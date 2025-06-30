#!/usr/bin/env python3
"""
Comprehensive test suite for Code Generator Agent
Tests all major functionality including code generation, quality analysis, and pattern learning
"""

import pytest
import asyncio
import ast
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import os

from core.code_generator_agent import (
    CodeGeneratorAgent,
    CodeSpecification,
    CodeType,
    Language,
    GeneratedCode,
    CodePattern,
    CodeQualityAnalyzer,
    MultiLLMOrchestrator,
    LLMInterface,
    FeedbackLoop,
    PatternLearner
)


# Test fixtures
@pytest.fixture
def quality_analyzer():
    """Create a CodeQualityAnalyzer instance"""
    return CodeQualityAnalyzer()


@pytest.fixture
def llm_orchestrator():
    """Create a MultiLLMOrchestrator instance"""
    return MultiLLMOrchestrator()


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider"""
    provider = Mock(spec=LLMInterface)
    provider.generate = AsyncMock(return_value="Generated code")
    provider.refine = AsyncMock(return_value="Refined code")
    return provider


@pytest.fixture
def code_spec():
    """Create a sample CodeSpecification"""
    return CodeSpecification(
        name="test_function",
        type=CodeType.FUNCTION,
        language=Language.PYTHON,
        description="A test function",
        requirements=["Handle errors", "Add logging"],
        inputs={"data": {"type": "str", "description": "Input data"}},
        outputs={"result": {"type": "bool", "description": "Success flag"}},
        patterns=["error_handling"],
        quality_requirements={"readability": 0.8, "testability": 0.7}
    )


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing"""
    return '''
def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

class DataProcessor:
    """Process data with validation"""
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        if not data:
            raise ValueError("No data provided")
        return [item * 2 for item in data]
'''


@pytest.fixture
async def code_generator():
    """Create a CodeGeneratorAgent instance with mocked components"""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = CodeGeneratorAgent(
            template_dir=Path(tmpdir),
            pattern_library_path=Path(tmpdir),
            enable_self_improvement=True
        )
        
        # Mock LLM provider
        mock_provider = Mock(spec=LLMInterface)
        mock_provider.generate = AsyncMock(return_value="def test(): pass")
        mock_provider.refine = AsyncMock(return_value="def test(): return True")
        
        agent.llm_orchestrator.register_provider("mock", mock_provider)
        
        yield agent


# Test CodeQualityAnalyzer
class TestCodeQualityAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_readability(self, quality_analyzer, sample_python_code):
        """Test readability analysis"""
        score = await quality_analyzer._analyze_readability(
            sample_python_code, Language.PYTHON
        )
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should have decent readability
    
    @pytest.mark.asyncio
    async def test_analyze_complexity(self, quality_analyzer):
        """Test complexity analysis"""
        simple_code = "def add(a, b): return a + b"
        complex_code = """
def complex_function(data):
    if data:
        for item in data:
            if item > 0:
                while item > 10:
                    item -= 1
                    if item % 2 == 0:
                        break
            else:
                try:
                    process(item)
                except:
                    pass
    return data
"""
        
        simple_score = await quality_analyzer._analyze_complexity(
            simple_code, Language.PYTHON
        )
        complex_score = await quality_analyzer._analyze_complexity(
            complex_code, Language.PYTHON
        )
        
        assert simple_score > complex_score
        assert simple_score > 0.8
        assert complex_score < 0.5
    
    @pytest.mark.asyncio
    async def test_analyze_security(self, quality_analyzer):
        """Test security analysis"""
        insecure_code = '''
password = "hardcoded123"
api_key = "secret_key_123"
eval(user_input)
os.system("rm -rf /")
'''
        secure_code = '''
def process_data(data):
    validated = validate_input(data)
    return transform(validated)
'''
        
        insecure_score = await quality_analyzer._analyze_security(
            insecure_code, Language.PYTHON
        )
        secure_score = await quality_analyzer._analyze_security(
            secure_code, Language.PYTHON
        )
        
        assert insecure_score < 0.3
        assert secure_score > 0.8
    
    @pytest.mark.asyncio
    async def test_analyze_testability(self, quality_analyzer):
        """Test testability analysis"""
        testable_code = '''
def pure_function(x, y):
    """Pure function with no side effects"""
    return x + y

class Service:
    def __init__(self, dependency):
        self.dependency = dependency
    
    def process(self, data):
        return self.dependency.transform(data)
'''
        
        untestable_code = '''
def bad_function():
    print("Side effect")
    global state
    state = "modified"
    open("/etc/passwd", "r")
'''
        
        testable_score = await quality_analyzer._analyze_testability(
            testable_code, Language.PYTHON
        )
        untestable_score = await quality_analyzer._analyze_testability(
            untestable_code, Language.PYTHON
        )
        
        assert testable_score > 0.7
        assert untestable_score < 0.5
    
    @pytest.mark.asyncio
    async def test_analyze_documentation(self, quality_analyzer):
        """Test documentation analysis"""
        documented_code = '''
def calculate_average(numbers):
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numeric values
        
    Returns:
        float: The average value
    """
    # Check for empty list
    if not numbers:
        return 0
    
    # Calculate sum and return average
    total = sum(numbers)
    return total / len(numbers)
'''
        
        undocumented_code = '''
def calc(n):
    if not n:
        return 0
    return sum(n) / len(n)
'''
        
        documented_score = await quality_analyzer._analyze_documentation(documented_code)
        undocumented_score = await quality_analyzer._analyze_documentation(undocumented_code)
        
        assert documented_score > 0.7
        assert undocumented_score < 0.3
    
    @pytest.mark.asyncio
    async def test_overall_analysis(self, quality_analyzer, sample_python_code):
        """Test complete code analysis"""
        metrics = await quality_analyzer.analyze(sample_python_code, Language.PYTHON)
        
        assert "overall" in metrics
        assert all(0 <= score <= 1 for score in metrics.values())
        assert len(metrics) >= 6  # Should have all metric types


# Test MultiLLMOrchestrator
class TestMultiLLMOrchestrator:
    def test_register_provider(self, llm_orchestrator, mock_llm_provider):
        """Test registering LLM providers"""
        llm_orchestrator.register_provider("test_provider", mock_llm_provider)
        
        assert "test_provider" in llm_orchestrator.providers
        assert "test_provider" in llm_orchestrator.performance_history
    
    @pytest.mark.asyncio
    async def test_generate_single_provider(self, llm_orchestrator, mock_llm_provider):
        """Test generation with single provider"""
        llm_orchestrator.register_provider("test", mock_llm_provider)
        
        result, provider = await llm_orchestrator.generate("Test prompt")
        
        assert result == "Generated code"
        assert provider == "test"
        mock_llm_provider.generate.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_ensemble_generation(self, llm_orchestrator):
        """Test ensemble generation strategy"""
        # Register multiple providers
        providers = []
        for i in range(3):
            provider = Mock(spec=LLMInterface)
            provider.generate = AsyncMock(return_value=f"Code version {i}" * (i + 1))
            llm_orchestrator.register_provider(f"provider_{i}", provider)
            providers.append(provider)
        
        result, provider = await llm_orchestrator.generate(
            "Test prompt", strategy="ensemble"
        )
        
        # Should return the longest result
        assert "Code version 2" in result
        assert provider.startswith("ensemble:")
        
        # All providers should be called
        for provider in providers:
            provider.generate.assert_called_once()
    
    def test_performance_tracking(self, llm_orchestrator, mock_llm_provider):
        """Test provider performance tracking"""
        llm_orchestrator.register_provider("test", mock_llm_provider)
        
        # Update performance
        llm_orchestrator.update_performance("test", 0.9)
        llm_orchestrator.update_performance("test", 0.8)
        llm_orchestrator.update_performance("test", 0.85)
        
        assert len(llm_orchestrator.performance_history["test"]) == 3
        assert llm_orchestrator.performance_history["test"][-1] == 0.85
    
    def test_select_best_performer(self, llm_orchestrator):
        """Test best performer selection"""
        # Register providers with different performance
        llm_orchestrator.register_provider("good", Mock(spec=LLMInterface))
        llm_orchestrator.register_provider("bad", Mock(spec=LLMInterface))
        
        llm_orchestrator.performance_history["good"] = [0.9, 0.95, 0.92]
        llm_orchestrator.performance_history["bad"] = [0.5, 0.6, 0.55]
        
        best = llm_orchestrator._select_best_performer()
        assert best == "good"


# Test CodeGeneratorAgent
class TestCodeGeneratorAgent:
    @pytest.mark.asyncio
    async def test_initialization(self, code_generator):
        """Test agent initialization"""
        assert code_generator.quality_analyzer is not None
        assert code_generator.llm_orchestrator is not None
        assert code_generator.enable_self_improvement is True
        assert hasattr(code_generator, 'feedback_loop')
        assert hasattr(code_generator, 'pattern_learner')
    
    @pytest.mark.asyncio
    async def test_generate_code_basic(self, code_generator, code_spec):
        """Test basic code generation"""
        # Mock quality analyzer
        code_generator.quality_analyzer.analyze = AsyncMock(
            return_value={
                "overall": 0.85,
                "readability": 0.9,
                "testability": 0.8,
                "security": 0.85
            }
        )
        
        result = await code_generator.generate_code(code_spec, iterations=1)
        
        assert isinstance(result, GeneratedCode)
        assert result.language == Language.PYTHON
        assert result.type == CodeType.FUNCTION
        assert result.quality_score == 0.85
        assert result.code is not None
    
    @pytest.mark.asyncio
    async def test_generate_code_with_refinement(self, code_generator, code_spec):
        """Test code generation with iterative refinement"""
        # Mock quality scores that improve over iterations
        quality_scores = [
            {"overall": 0.6, "readability": 0.5},
            {"overall": 0.75, "readability": 0.7},
            {"overall": 0.85, "readability": 0.9}
        ]
        
        code_generator.quality_analyzer.analyze = AsyncMock(
            side_effect=quality_scores
        )
        
        result = await code_generator.generate_code(
            code_spec, 
            iterations=3,
            quality_threshold=0.8
        )
        
        assert result.quality_score == 0.85
        assert code_generator.quality_analyzer.analyze.call_count <= 3
    
    @pytest.mark.asyncio
    async def test_pattern_finding(self, code_generator, code_spec):
        """Test finding relevant patterns"""
        # Add test patterns
        test_pattern = CodePattern(
            name="error_handling",
            description="Standard error handling pattern",
            template="try: ... except Exception as e: ...",
            language=Language.PYTHON,
            parameters={},
            tags={"error", "exception"}
        )
        code_generator.pattern_library = {"error_handling": test_pattern}
        
        patterns = await code_generator._find_relevant_patterns(code_spec)
        
        assert len(patterns) > 0
        assert patterns[0].name == "error_handling"
    
    @pytest.mark.asyncio
    async def test_format_python_code(self, code_generator):
        """Test Python code formatting"""
        unformatted = "def test(  x,y  ):return x+y"
        
        with patch('black.format_str') as mock_black:
            mock_black.return_value = "def test(x, y):\n    return x + y"
            
            formatted = await code_generator._format_python(unformatted)
            
            assert "return x + y" in formatted
            mock_black.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_dependencies(self, code_generator):
        """Test dependency extraction"""
        code = '''
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List
import os
'''
        
        deps = await code_generator._extract_dependencies(code, Language.PYTHON)
        
        assert "numpy" in deps
        assert "sklearn" in deps
        assert "pandas" in deps
        assert "typing" not in deps  # Standard library
        assert "os" not in deps  # Standard library
    
    @pytest.mark.asyncio
    async def test_generate_tests(self, code_generator, code_spec):
        """Test test generation"""
        sample_code = "def add(a, b): return a + b"
        
        # Mock LLM to return test code
        test_code = '''
def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
'''
        code_generator.llm_orchestrator.generate = AsyncMock(
            return_value=(test_code, "mock")
        )
        
        tests = await code_generator._generate_tests(sample_code, code_spec)
        
        assert "test_add" in tests
        assert "assert" in tests
    
    @pytest.mark.asyncio
    async def test_generate_documentation(self, code_generator, code_spec):
        """Test documentation generation"""
        sample_code = "def process(data): return data"
        
        doc = '''
# Process Function

## Overview
Processes input data

## Usage
```python
result = process(data)
```
'''
        code_generator.llm_orchestrator.generate = AsyncMock(
            return_value=(doc, "mock")
        )
        
        documentation = await code_generator._generate_documentation(
            sample_code, code_spec
        )
        
        assert "# Process Function" in documentation
        assert "## Usage" in documentation
    
    @pytest.mark.asyncio
    async def test_improve_existing_code(self, code_generator):
        """Test code improvement functionality"""
        original_code = '''
def process(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
        
        improved_code = '''
def process(data: List[int]) -> List[int]:
    """Process data by doubling each item."""
    return [item * 2 for item in data]
'''
        
        code_generator.llm_orchestrator.generate = AsyncMock(
            return_value=(improved_code, "mock")
        )
        code_generator.quality_analyzer.analyze = AsyncMock(
            side_effect=[
                {"overall": 0.6},  # Original score
                {"overall": 0.9}   # Improved score
            ]
        )
        
        result = await code_generator.improve_existing_code(
            original_code,
            Language.PYTHON,
            ["Add type hints", "Improve performance"]
        )
        
        assert result.quality_score == 0.9
        assert "List[int]" in result.code
    
    @pytest.mark.asyncio
    async def test_generate_from_examples(self, code_generator):
        """Test generation from examples"""
        examples = [
            {
                "code": "def multiply(x, y): return x * y",
                "input": "multiply(3, 4)",
                "output": "12"
            },
            {
                "code": "def divide(x, y): return x / y if y != 0 else None",
                "input": "divide(10, 2)",
                "output": "5.0"
            }
        ]
        
        # Mock pattern extraction and generation
        code_generator._extract_patterns_from_examples = AsyncMock(
            return_value=[{"structure": {"functions": ["multiply", "divide"]}}]
        )
        code_generator.generate_code = AsyncMock(
            return_value=GeneratedCode(
                code="def calculate(x, y, op): ...",
                language=Language.PYTHON,
                type=CodeType.FUNCTION,
                quality_score=0.85
            )
        )
        
        result = await code_generator.generate_from_examples(
            examples,
            "Create a calculator function",
            Language.PYTHON
        )
        
        assert result.quality_score == 0.85
        code_generator._extract_patterns_from_examples.assert_called_once()


# Test FeedbackLoop
class TestFeedbackLoop:
    @pytest.mark.asyncio
    async def test_process_feedback(self, code_spec):
        """Test feedback processing"""
        feedback_loop = FeedbackLoop()
        
        result = GeneratedCode(
            code="test code",
            language=Language.PYTHON,
            type=CodeType.FUNCTION,
            quality_score=0.7,
            metrics={"readability": 0.6, "performance": 0.8}
        )
        
        await feedback_loop.process_feedback(
            code_spec,
            result,
            "Needs better error handling"
        )
        
        assert len(feedback_loop.feedback_history) == 1
        entry = feedback_loop.feedback_history[0]
        assert entry["user_feedback"] == "Needs better error handling"
        assert "Focus on error handling" in entry["lessons"]
    
    @pytest.mark.asyncio
    async def test_extract_lessons(self):
        """Test lesson extraction from feedback"""
        feedback_loop = FeedbackLoop()
        
        spec = Mock()
        spec.language = Language.PYTHON
        spec.type = CodeType.FUNCTION
        
        result = Mock()
        result.metrics = {
            "readability": 0.5,
            "performance": 0.9,
            "security": 0.6
        }
        
        lessons = await feedback_loop._extract_lessons(
            spec,
            result,
            "Code is hard to understand and has security issues"
        )
        
        assert any("readability" in lesson for lesson in lessons)
        assert any("security" in lesson for lesson in lessons)
        assert any("readable" in lesson for lesson in lessons)


# Test PatternLearner
class TestPatternLearner:
    @pytest.mark.asyncio
    async def test_learn_from_generation(self, code_spec):
        """Test pattern learning from successful generation"""
        pattern_learner = PatternLearner()
        
        result = GeneratedCode(
            code='''
@property
def value(self):
    return self._value

@staticmethod
def create():
    return MyClass()
''',
            language=Language.PYTHON,
            type=CodeType.CLASS,
            quality_score=0.95
        )
        
        await pattern_learner.learn_from_generation(code_spec, result)
        
        key = f"{Language.PYTHON.value}:function"
        assert key in pattern_learner.learned_patterns
        assert len(pattern_learner.learned_patterns[key]) > 0
    
    @pytest.mark.asyncio
    async def test_extract_patterns(self):
        """Test pattern extraction from code"""
        pattern_learner = PatternLearner()
        
        code = '''
class DataProcessor:
    """Process data with validation"""
    
    def __init__(self, config):
        self.config = config
    
    @property
    def is_configured(self):
        return self.config is not None
    
    @staticmethod
    def validate(data):
        return data is not None
    
    async def process(self, data):
        if self.validate(data):
            return await self._process_internal(data)
'''
        
        patterns = await pattern_learner._extract_patterns(code, Language.PYTHON)
        
        assert len(patterns) > 0
        
        # Check for class pattern
        class_patterns = [p for p in patterns if p["type"] == "class"]
        assert len(class_patterns) == 1
        assert "DataProcessor" in class_patterns[0]["name"]
        
        # Check for function patterns
        func_patterns = [p for p in patterns if p["type"] == "function"]
        assert any(p["name"] == "is_configured" for p in func_patterns)
        assert any("property" in p["decorators"] for p in func_patterns)
        assert any("staticmethod" in p["decorators"] for p in func_patterns)


# Integration tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_generation_pipeline(self, code_generator):
        """Test complete code generation pipeline"""
        spec = CodeSpecification(
            name="data_validator",
            type=CodeType.CLASS,
            language=Language.PYTHON,
            description="A data validation class with type checking",
            requirements=[
                "Validate different data types",
                "Provide detailed error messages",
                "Support custom validators"
            ],
            inputs={
                "data": {"type": "Any", "description": "Data to validate"},
                "schema": {"type": "Dict", "description": "Validation schema"}
            },
            outputs={
                "is_valid": {"type": "bool", "description": "Validation result"},
                "errors": {"type": "List[str]", "description": "Validation errors"}
            },
            quality_requirements={
                "testability": 0.8,
                "readability": 0.85
            }
        )
        
        # Mock responses
        generated_code = '''
class DataValidator:
    """Validates data against schemas."""
    
    def __init__(self, schema):
        self.schema = schema
    
    def validate(self, data):
        """Validate data against schema."""
        errors = []
        # Validation logic here
        return len(errors) == 0, errors
'''
        
        test_code = '''
def test_data_validator():
    validator = DataValidator({"type": "string"})
    assert validator.validate("test")[0] == True
'''
        
        documentation = '''
# DataValidator

A flexible data validation class.

## Usage
```python
validator = DataValidator(schema)
is_valid, errors = validator.validate(data)
```
'''
        
        # Setup mocks
        code_generator.llm_orchestrator.generate = AsyncMock(
            side_effect=[
                (generated_code, "mock"),  # Initial generation
                (test_code, "mock"),       # Test generation
                (documentation, "mock")     # Documentation generation
            ]
        )
        
        code_generator.quality_analyzer.analyze = AsyncMock(
            return_value={
                "overall": 0.9,
                "readability": 0.85,
                "testability": 0.85,
                "security": 0.9,
                "performance": 0.8,
                "documentation": 0.7
            }
        )
        
        # Generate code
        result = await code_generator.generate_code(spec)
        
        # Verify result
        assert result.quality_score >= 0.85
        assert "DataValidator" in result.code
        assert result.tests is not None
        assert "test_data_validator" in result.tests
        assert result.documentation is not None
        assert "# DataValidator" in result.documentation
        assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_self_improvement_cycle(self, code_generator):
        """Test self-improvement through feedback"""
        # Generate initial code
        spec = CodeSpecification(
            name="calculator",
            type=CodeType.FUNCTION,
            language=Language.PYTHON,
            description="Basic calculator",
            requirements=["Add two numbers"],
            inputs={"a": {"type": "int"}, "b": {"type": "int"}},
            outputs={"sum": {"type": "int"}}
        )
        
        code_generator.quality_analyzer.analyze = AsyncMock(
            return_value={"overall": 0.75}
        )
        
        result = await code_generator.generate_code(spec)
        
        # Process feedback
        await code_generator.feedback_loop.process_feedback(
            spec,
            result,
            "Add input validation"
        )
        
        # Verify feedback was recorded
        assert len(code_generator.feedback_loop.feedback_history) == 1
        
        # Verify pattern learning (if quality was high enough)
        if result.quality_score > 0.9:
            assert len(code_generator.pattern_learner.learned_patterns) > 0


# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_specification(self, code_generator):
        """Test handling of minimal specification"""
        minimal_spec = CodeSpecification(
            name="minimal",
            type=CodeType.FUNCTION,
            language=Language.PYTHON,
            description="Minimal function",
            requirements=[],
            inputs={},
            outputs={}
        )
        
        code_generator.quality_analyzer.analyze = AsyncMock(
            return_value={"overall": 0.7}
        )
        
        result = await code_generator.generate_code(minimal_spec)
        assert result is not None
        assert result.code is not None
    
    @pytest.mark.asyncio
    async def test_llm_failure_handling(self, code_generator):
        """Test handling of LLM failures"""
        spec = Mock()
        spec.name = "test"
        spec.type = CodeType.FUNCTION
        spec.language = Language.PYTHON
        
        # Make LLM fail
        code_generator.llm_orchestrator.generate = AsyncMock(
            side_effect=Exception("LLM error")
        )
        
        with pytest.raises(Exception):
            await code_generator.generate_code(spec)
    
    @pytest.mark.asyncio
    async def test_large_code_analysis(self, quality_analyzer):
        """Test analysis of large code files"""
        # Generate large code
        large_code = "\n".join([
            f"def function_{i}(x):\n    return x * {i}"
            for i in range(100)
        ])
        
        metrics = await quality_analyzer.analyze(large_code, Language.PYTHON)
        
        assert metrics is not None
        assert "overall" in metrics
    
    def test_pattern_library_loading(self):
        """Test pattern library loading from files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern_dir = Path(tmpdir)
            
            # Create test pattern file
            pattern_file = pattern_dir / "test_pattern.yaml"
            pattern_file.write_text('''
name: test_pattern
description: Test pattern
template: "def test(): pass"
language: python
parameters: {}
''')
            
            agent = CodeGeneratorAgent(pattern_library_path=pattern_dir)
            
            assert len(agent.pattern_library) == 1
            assert "test_pattern" in agent.pattern_library


if __name__ == "__main__":
    pytest.main([__file__, "-v"])