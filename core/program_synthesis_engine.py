#!/usr/bin/env python3
"""
Program Synthesis Engine for JARVIS
Advanced code generation with multiple synthesis strategies,
semantic caching, and quality control.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis
from collections import OrderedDict
import structlog
import ast
import inspect
import textwrap

from .config_manager import config_manager, get_synthesis_config
from .monitoring import monitor_performance, monitoring_service

logger = structlog.get_logger()


@dataclass
class SynthesisRequest:
    """Request for program synthesis"""
    description: str
    language: str = "python"
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class SynthesisResult:
    """Result of program synthesis"""
    code: str
    method: str
    confidence: float
    tests: List[str]
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    synthesis_time: float = 0.0


class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()


class SemanticCache:
    """Semantic caching with similarity matching"""
    
    def __init__(self, threshold: float = 0.85, max_size: int = 1000):
        self.threshold = threshold
        self.max_size = max_size
        self.cache = []
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.vectors = None
        self.redis_client = None
        
        # Try to connect to Redis for persistent cache
        try:
            redis_config = config_manager.get_redis_config()
            self.redis_client = redis.Redis(**redis_config)
            self.redis_client.ping()
            logger.info("Connected to Redis for semantic cache")
        except Exception as e:
            logger.warning(f"Redis not available for semantic cache: {e}")
    
    def _compute_similarity(self, query: str) -> Tuple[int, float]:
        """Compute similarity with cached items"""
        if not self.cache:
            return -1, 0.0
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return best_idx, best_score
    
    def get(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Get cached result if similar request exists"""
        query = f"{request.description} {json.dumps(request.constraints)}"
        
        # Check Redis first if available
        if self.redis_client:
            try:
                key = f"synthesis:semantic:{hashlib.md5(query.encode()).hexdigest()}"
                cached = self.redis_client.get(key)
                if cached:
                    return SynthesisResult(**json.loads(cached))
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
        
        # Check in-memory cache
        if not self.cache:
            return None
        
        best_idx, score = self._compute_similarity(query)
        
        if score >= self.threshold:
            logger.info(f"Semantic cache hit with score {score:.3f}")
            return self.cache[best_idx][1]
        
        return None
    
    def put(self, request: SynthesisRequest, result: SynthesisResult):
        """Add result to cache"""
        query = f"{request.description} {json.dumps(request.constraints)}"
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"synthesis:semantic:{hashlib.md5(query.encode()).hexdigest()}"
                self.redis_client.setex(
                    key,
                    timedelta(seconds=3600),
                    json.dumps(result.__dict__)
                )
            except Exception as e:
                logger.error(f"Redis cache write error: {e}")
        
        # Add to in-memory cache
        self.cache.append((query, result))
        
        # Limit cache size
        if len(self.cache) > self.max_size:
            self.cache = self.cache[-self.max_size:]
        
        # Update vectorizer
        queries = [item[0] for item in self.cache]
        self.vectors = self.vectorizer.fit_transform(queries)


class SynthesisStrategy:
    """Base class for synthesis strategies"""
    
    async def synthesize(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        raise NotImplementedError


class PatternBasedStrategy(SynthesisStrategy):
    """Pattern-based synthesis using code templates"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, str]:
        """Load code patterns"""
        return {
            "filter": """
def {func_name}({params}):
    \"\"\"Filter {description}\"\"\"
    return [item for item in {input_var} if {condition}]
""",
            "map": """
def {func_name}({params}):
    \"\"\"Map {description}\"\"\"
    return [{transform} for item in {input_var}]
""",
            "reduce": """
def {func_name}({params}):
    \"\"\"Reduce {description}\"\"\"
    result = {initial}
    for item in {input_var}:
        result = {operation}
    return result
""",
            "search": """
def {func_name}({params}):
    \"\"\"Search for {description}\"\"\"
    for i, item in enumerate({input_var}):
        if {condition}:
            return i
    return -1
""",
            "sort": """
def {func_name}({params}):
    \"\"\"Sort {description}\"\"\"
    return sorted({input_var}, key=lambda x: {key})
"""
        }
    
    async def synthesize(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Synthesize using patterns"""
        # Analyze request to determine pattern
        description = request.description.lower()
        
        pattern_type = None
        if any(word in description for word in ["filter", "select", "find all"]):
            pattern_type = "filter"
        elif any(word in description for word in ["map", "transform", "convert"]):
            pattern_type = "map"
        elif any(word in description for word in ["reduce", "aggregate", "sum", "count"]):
            pattern_type = "reduce"
        elif any(word in description for word in ["search", "find", "locate"]):
            pattern_type = "search"
        elif any(word in description for word in ["sort", "order"]):
            pattern_type = "sort"
        
        if not pattern_type:
            return None
        
        # Generate code from pattern
        template = self.patterns[pattern_type]
        
        # Extract parameters from description
        func_name = "synthesized_function"
        params = "data"
        input_var = "data"
        
        # Pattern-specific generation
        if pattern_type == "filter":
            condition = "True"  # Would be extracted from NLP
            code = template.format(
                func_name=func_name,
                params=params,
                description=description,
                input_var=input_var,
                condition=condition
            )
        else:
            # Similar for other patterns
            code = template.format(
                func_name=func_name,
                params=params,
                description=description,
                input_var=input_var
            )
        
        # Generate tests
        tests = self._generate_tests(func_name, pattern_type)
        
        return SynthesisResult(
            code=code.strip(),
            method="pattern_based",
            confidence=0.7,
            tests=tests,
            explanation=f"Generated using {pattern_type} pattern",
            metadata={"pattern": pattern_type}
        )
    
    def _generate_tests(self, func_name: str, pattern_type: str) -> List[str]:
        """Generate tests for synthesized function"""
        tests = []
        
        if pattern_type == "filter":
            tests.append(f"""
def test_{func_name}_empty():
    assert {func_name}([]) == []

def test_{func_name}_basic():
    data = [1, 2, 3, 4, 5]
    result = {func_name}(data)
    assert isinstance(result, list)
""")
        
        return tests


class TemplateBasedStrategy(SynthesisStrategy):
    """Template-based synthesis with placeholders"""
    
    async def synthesize(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Synthesize using templates"""
        # Implementation would use more sophisticated template matching
        return None


class NeuralStrategy(SynthesisStrategy):
    """Neural synthesis using ML models"""
    
    async def synthesize(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Synthesize using neural models"""
        # Would integrate with actual ML models
        return None


class ExampleBasedStrategy(SynthesisStrategy):
    """Synthesis from input-output examples"""
    
    async def synthesize(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Synthesize from examples"""
        if not request.examples:
            return None
        
        # Analyze examples to infer function
        # This is a simplified implementation
        return None


class ConstraintBasedStrategy(SynthesisStrategy):
    """Synthesis using constraint solving"""
    
    async def synthesize(self, request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Synthesize using constraints"""
        if not request.constraints:
            return None
        
        # Would use constraint solver
        return None


class ProgramSynthesisEngine:
    """Main program synthesis engine"""
    
    def __init__(self):
        self.config = get_synthesis_config()
        self.lru_cache = LRUCache(max_size=self.config.cache_max_size)
        self.semantic_cache = SemanticCache(
            threshold=self.config.semantic_cache_threshold,
            max_size=self.config.cache_max_size
        )
        
        # Initialize strategies
        self.strategies = {
            "pattern_based": PatternBasedStrategy(),
            "template_based": TemplateBasedStrategy(),
            "neural": NeuralStrategy(),
            "example_based": ExampleBasedStrategy(),
            "constraint_based": ConstraintBasedStrategy()
        }
        
        # Strategy weights for ensemble
        self.strategy_weights = {
            "pattern_based": 1.0,
            "template_based": 0.8,
            "neural": 0.9,
            "example_based": 0.7,
            "constraint_based": 0.6
        }
    
    @monitor_performance("synthesis_engine")
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Main synthesis method"""
        start_time = time.time()
        
        # Check caches
        cache_key = self._get_cache_key(request)
        
        # Check LRU cache first
        if self.config.enable_cache:
            cached = self.lru_cache.get(cache_key)
            if cached:
                logger.info("LRU cache hit")
                monitoring_service.metrics_collector.record_event({
                    "event_type": "cache_hit",
                    "cache_type": "lru"
                })
                return cached
        
        # Check semantic cache
        if self.config.semantic_cache_enabled:
            cached = self.semantic_cache.get(request)
            if cached:
                logger.info("Semantic cache hit")
                monitoring_service.metrics_collector.record_event({
                    "event_type": "cache_hit",
                    "cache_type": "semantic"
                })
                return cached
        
        # Try synthesis strategies
        results = await self._try_strategies(request)
        
        if not results:
            # Fallback result
            result = SynthesisResult(
                code="# Unable to synthesize function",
                method="none",
                confidence=0.0,
                tests=[],
                explanation="No synthesis strategy succeeded"
            )
        else:
            # Select best result
            result = self._select_best_result(results)
        
        # Calculate synthesis time
        result.synthesis_time = time.time() - start_time
        
        # Cache result
        if self.config.enable_cache and result.confidence > 0.5:
            self.lru_cache.put(cache_key, result)
            
            if self.config.semantic_cache_enabled:
                self.semantic_cache.put(request, result)
        
        # Log metrics
        monitoring_service.metrics_collector.record_event({
            "event_type": "synthesis_completed",
            "method": result.method,
            "confidence": result.confidence,
            "duration": result.synthesis_time
        })
        
        return result
    
    async def _try_strategies(self, request: SynthesisRequest) -> List[SynthesisResult]:
        """Try multiple synthesis strategies"""
        tasks = []
        
        for name, strategy in self.strategies.items():
            if name in self.config.synthesis_methods:
                task = self._try_strategy(name, strategy, request)
                tasks.append(task)
        
        # Run strategies concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        valid_results = []
        for result in results:
            if isinstance(result, SynthesisResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Strategy failed: {result}")
        
        return valid_results
    
    async def _try_strategy(self, name: str, strategy: SynthesisStrategy, 
                          request: SynthesisRequest) -> Optional[SynthesisResult]:
        """Try a single strategy with timeout"""
        try:
            result = await asyncio.wait_for(
                strategy.synthesize(request),
                timeout=self.config.max_synthesis_time
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Strategy {name} timed out")
            return None
        except Exception as e:
            logger.error(f"Strategy {name} failed: {e}")
            return None
    
    def _select_best_result(self, results: List[SynthesisResult]) -> SynthesisResult:
        """Select best result from multiple strategies"""
        if not results:
            return None
        
        # Score each result
        scores = []
        for result in results:
            score = result.confidence * self.strategy_weights.get(result.method, 0.5)
            
            # Bonus for having tests
            if result.tests:
                score *= 1.2
            
            # Penalty for empty code
            if not result.code or result.code.strip() == "":
                score *= 0.1
            
            scores.append(score)
        
        # Return result with highest score
        best_idx = np.argmax(scores)
        return results[best_idx]
    
    def _get_cache_key(self, request: SynthesisRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "description": request.description,
            "language": request.language,
            "constraints": request.constraints,
            "examples": request.examples
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return {
            "lru_cache_size": len(self.lru_cache.cache),
            "semantic_cache_size": len(self.semantic_cache.cache),
            "available_strategies": list(self.strategies.keys()),
            "enabled_strategies": self.config.synthesis_methods
        }


# Global synthesis engine instance
synthesis_engine = ProgramSynthesisEngine()


# Convenience function for JARVIS integration
async def synthesize_function(description: str, **kwargs) -> SynthesisResult:
    """Synthesize a function from description"""
    request = SynthesisRequest(
        description=description,
        **kwargs
    )
    return await synthesis_engine.synthesize(request)