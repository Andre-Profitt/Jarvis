"""
World-Class Code Generator Agent
================================

An advanced AI-powered code generation agent that:
- Generates high-quality, production-ready code
- Understands and implements complex architectures
- Self-improves through feedback loops
- Integrates with multiple LLMs for best results
- Performs automatic testing and validation
- Supports multiple programming languages
- Implements design patterns and best practices
"""

import ast
import asyncio
import json
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import yaml

import black
import isort
from jinja2 import Environment, FileSystemLoader, Template
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge
import docker
import git
import aiohttp
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = get_logger(__name__)

# Metrics
code_generations = Counter(
    "code_generations_total", "Total code generations", ["language", "type"]
)
generation_time = Histogram("code_generation_duration_seconds", "Code generation time")
generation_quality = Gauge(
    "code_generation_quality_score", "Quality score of generated code"
)
test_pass_rate = Gauge(
    "generated_code_test_pass_rate", "Test pass rate for generated code"
)


class CodeType(Enum):
    """Types of code that can be generated"""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    API = "api"
    TEST = "test"
    DOCUMENTATION = "documentation"
    SCRIPT = "script"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"


class Language(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    SWIFT = "swift"


@dataclass
class CodeSpecification:
    """Detailed specification for code generation"""

    name: str
    type: CodeType
    language: Language
    description: str
    requirements: List[str]
    inputs: Dict[str, Dict[str, Any]]  # param -> {type, description, required}
    outputs: Dict[str, Dict[str, Any]]  # return -> {type, description}
    dependencies: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)  # Design patterns to use
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, str]] = field(default_factory=list)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedCode:
    """Container for generated code and metadata"""

    code: str
    language: Language
    type: CodeType
    tests: Optional[str] = None
    documentation: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)
    generation_time: float = 0.0
    model_used: str = ""
    confidence: float = 0.0


@dataclass
class CodePattern:
    """Reusable code pattern"""

    name: str
    description: str
    template: str
    language: Language
    parameters: Dict[str, Any]
    examples: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)


class CodeQualityAnalyzer:
    """Analyzes code quality metrics"""

    def __init__(self):
        self.metrics_weights = {
            "readability": 0.2,
            "maintainability": 0.2,
            "performance": 0.15,
            "security": 0.15,
            "testability": 0.15,
            "documentation": 0.15,
        }

    async def analyze(self, code: str, language: Language) -> Dict[str, float]:
        """Comprehensive code quality analysis"""
        metrics = {}

        # Readability analysis
        metrics["readability"] = await self._analyze_readability(code, language)

        # Complexity analysis
        metrics["maintainability"] = await self._analyze_complexity(code, language)

        # Security analysis
        metrics["security"] = await self._analyze_security(code, language)

        # Performance analysis
        metrics["performance"] = await self._analyze_performance(code, language)

        # Test coverage potential
        metrics["testability"] = await self._analyze_testability(code, language)

        # Documentation quality
        metrics["documentation"] = await self._analyze_documentation(code)

        # Calculate overall score
        overall_score = sum(
            metrics.get(metric, 0) * weight
            for metric, weight in self.metrics_weights.items()
        )

        metrics["overall"] = overall_score
        return metrics

    async def _analyze_readability(self, code: str, language: Language) -> float:
        """Analyze code readability"""
        score = 1.0

        # Check line length
        lines = code.split("\n")
        long_lines = sum(1 for line in lines if len(line) > 80)
        score -= (long_lines / max(len(lines), 1)) * 0.3

        # Check naming conventions
        if language == Language.PYTHON:
            # Check for PEP 8 compliance
            snake_case_pattern = re.compile(r"^[a-z_][a-z0-9_]*$")
            class_pattern = re.compile(r"^[A-Z][a-zA-Z0-9]*$")

            # Simple heuristic for Python naming
            if "def " in code:
                functions = re.findall(r"def\s+(\w+)", code)
                valid_names = sum(1 for f in functions if snake_case_pattern.match(f))
                score -= (1 - valid_names / max(len(functions), 1)) * 0.2

        # Check for adequate spacing
        if "\n\n" not in code:
            score -= 0.1  # Penalty for no paragraph breaks

        return max(0, min(1, score))

    async def _analyze_complexity(self, code: str, language: Language) -> float:
        """Analyze code complexity"""
        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)
                complexity = self._calculate_cyclomatic_complexity(tree)
                # Convert to 0-1 score (lower complexity is better)
                return max(0, 1 - (complexity - 1) / 20)
            except:
                return 0.5

        # Default for other languages
        return 0.7

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for Python AST"""
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += len(node.items)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    async def _analyze_security(self, code: str, language: Language) -> float:
        """Analyze security aspects"""
        score = 1.0

        # Check for common security issues
        security_issues = [
            (r"eval\s*\(", 0.3),  # eval usage
            (r"exec\s*\(", 0.3),  # exec usage
            (r"pickle\.loads", 0.2),  # unsafe deserialization
            (r"os\.system", 0.2),  # command injection risk
            (r"shell\s*=\s*True", 0.2),  # shell injection risk
            (r'password\s*=\s*["\']\w+["\']', 0.4),  # hardcoded password
            (r'api_key\s*=\s*["\']\w+["\']', 0.4),  # hardcoded API key
        ]

        for pattern, penalty in security_issues:
            if re.search(pattern, code, re.IGNORECASE):
                score -= penalty

        return max(0, score)

    async def _analyze_performance(self, code: str, language: Language) -> float:
        """Analyze performance characteristics"""
        score = 1.0

        # Check for common performance issues
        if language == Language.PYTHON:
            # Nested loops
            nested_loops = len(re.findall(r"for .* in .*:\s*\n\s*for", code))
            score -= nested_loops * 0.1

            # Repeated concatenation in loops
            if re.search(r"for .* in .*:[\s\S]*?\+=", code):
                score -= 0.1

        return max(0, min(1, score))

    async def _analyze_testability(self, code: str, language: Language) -> float:
        """Analyze how testable the code is"""
        score = 0.5  # Base score

        # Check for proper function/method structure
        if language == Language.PYTHON:
            functions = len(re.findall(r"def \w+\(", code))
            classes = len(re.findall(r"class \w+", code))

            if functions > 0 or classes > 0:
                score += 0.2

            # Check for dependency injection patterns
            if "__init__" in code and "self." in code:
                score += 0.1

            # Check for pure functions (no side effects indicators)
            if not re.search(r"(print|open|write|global)", code):
                score += 0.2

        return min(1, score)

    async def _analyze_documentation(self, code: str) -> float:
        """Analyze documentation quality"""
        lines = code.split("\n")
        total_lines = len(lines)

        # Count docstrings and comments
        docstring_lines = len(re.findall(r'"""[\s\S]*?"""', code, re.MULTILINE))
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))

        doc_ratio = (docstring_lines * 3 + comment_lines) / max(total_lines, 1)

        return min(1, doc_ratio * 2)  # Scale up, cap at 1


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate code from prompt"""
        pass

    @abstractmethod
    async def refine(self, code: str, feedback: str, **kwargs) -> str:
        """Refine existing code based on feedback"""
        pass


class MultiLLMOrchestrator:
    """Orchestrates multiple LLMs for best results"""

    def __init__(self):
        self.providers = {}
        self.performance_history = {}
        self.selection_strategy = "weighted_random"  # or "best_performer", "ensemble"

    def register_provider(self, name: str, provider: LLMInterface):
        """Register an LLM provider"""
        self.providers[name] = provider
        self.performance_history[name] = []

    async def generate(
        self, prompt: str, strategy: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate code using selected strategy"""
        strategy = strategy or self.selection_strategy

        if strategy == "ensemble":
            return await self._ensemble_generate(prompt)
        elif strategy == "best_performer":
            provider_name = self._select_best_performer()
            result = await self.providers[provider_name].generate(prompt)
            return result, provider_name
        else:  # weighted_random
            provider_name = self._select_weighted_random()
            result = await self.providers[provider_name].generate(prompt)
            return result, provider_name

    async def _ensemble_generate(self, prompt: str) -> Tuple[str, str]:
        """Generate using ensemble of models"""
        results = []

        # Generate from all providers
        for name, provider in self.providers.items():
            try:
                result = await provider.generate(prompt)
                results.append((name, result))
            except Exception as e:
                logger.warning(f"Provider {name} failed", error=str(e))

        if not results:
            raise ValueError("All providers failed")

        # Select best result (simplified - could use voting or merging)
        # For now, return the longest result as a heuristic
        best_result = max(results, key=lambda x: len(x[1]))
        return best_result[1], f"ensemble:{best_result[0]}"

    def _select_best_performer(self) -> str:
        """Select provider with best historical performance"""
        if not self.performance_history:
            return list(self.providers.keys())[0]

        avg_scores = {}
        for name, history in self.performance_history.items():
            if history:
                avg_scores[name] = np.mean(history[-10:])  # Recent performance

        if not avg_scores:
            return list(self.providers.keys())[0]

        return max(avg_scores, key=avg_scores.get)

    def _select_weighted_random(self) -> str:
        """Select provider using weighted random based on performance"""
        names = list(self.providers.keys())
        if len(names) == 1:
            return names[0]

        # Calculate weights based on performance
        weights = []
        for name in names:
            history = self.performance_history.get(name, [])
            if history:
                weight = np.mean(history[-10:]) + 0.1  # Add small base weight
            else:
                weight = 0.5  # Default weight for new providers
            weights.append(weight)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        return np.random.choice(names, p=weights)

    def update_performance(self, provider_name: str, score: float):
        """Update provider performance history"""
        if provider_name.startswith("ensemble:"):
            provider_name = provider_name.split(":")[1]

        if provider_name in self.performance_history:
            self.performance_history[provider_name].append(score)


class CodeGeneratorAgent:
    """
    World-class code generation agent with advanced capabilities
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        pattern_library_path: Optional[Path] = None,
        enable_self_improvement: bool = True,
    ):

        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.pattern_library_path = (
            pattern_library_path or Path(__file__).parent / "patterns"
        )
        self.enable_self_improvement = enable_self_improvement

        # Initialize components
        self.quality_analyzer = CodeQualityAnalyzer()
        self.llm_orchestrator = MultiLLMOrchestrator()
        self.pattern_library = self._load_pattern_library()
        self.generation_history = []

        # Template engine
        self.template_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Code formatters
        self.formatters = {
            Language.PYTHON: self._format_python,
            Language.JAVASCRIPT: self._format_javascript,
            Language.TYPESCRIPT: self._format_typescript,
            Language.GO: self._format_go,
        }

        # Self-improvement components
        if enable_self_improvement:
            self.feedback_loop = FeedbackLoop()
            self.pattern_learner = PatternLearner()

        logger.info(
            "Code Generator Agent initialized",
            template_dir=str(self.template_dir),
            patterns_loaded=len(self.pattern_library),
        )

    async def generate_code(
        self,
        specification: CodeSpecification,
        iterations: int = 3,
        quality_threshold: float = 0.8,
    ) -> GeneratedCode:
        """
        Generate high-quality code from specification

        Args:
            specification: Detailed code specification
            iterations: Number of refinement iterations
            quality_threshold: Minimum quality score required

        Returns:
            Generated code with metadata
        """
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(
                "Starting code generation",
                name=specification.name,
                type=specification.type.value,
                language=specification.language.value,
            )

            # Phase 1: Pattern matching
            relevant_patterns = await self._find_relevant_patterns(specification)

            # Phase 2: Initial generation
            prompt = await self._create_generation_prompt(
                specification, relevant_patterns
            )
            initial_code, model_used = await self.llm_orchestrator.generate(prompt)

            # Phase 3: Iterative refinement
            refined_code = initial_code
            quality_metrics = {}

            for iteration in range(iterations):
                # Analyze quality
                quality_metrics = await self.quality_analyzer.analyze(
                    refined_code, specification.language
                )

                if quality_metrics["overall"] >= quality_threshold:
                    break

                # Generate improvement feedback
                feedback = await self._generate_improvement_feedback(
                    refined_code, quality_metrics, specification
                )

                # Refine code
                refined_code = await self._refine_code(
                    refined_code, feedback, specification
                )

            # Phase 4: Generate tests
            tests = await self._generate_tests(refined_code, specification)

            # Phase 5: Generate documentation
            documentation = await self._generate_documentation(
                refined_code, specification
            )

            # Phase 6: Format and finalize
            final_code = await self._format_code(refined_code, specification.language)

            # Phase 7: Extract dependencies
            dependencies = await self._extract_dependencies(
                final_code, specification.language
            )

            # Create result
            generation_time = asyncio.get_event_loop().time() - start_time

            result = GeneratedCode(
                code=final_code,
                language=specification.language,
                type=specification.type,
                tests=tests,
                documentation=documentation,
                dependencies=dependencies,
                metrics=quality_metrics,
                quality_score=quality_metrics.get("overall", 0),
                generation_time=generation_time,
                model_used=model_used,
                confidence=self._calculate_confidence(quality_metrics, iterations),
            )

            # Update metrics
            code_generations.labels(
                language=specification.language.value, type=specification.type.value
            ).inc()
            generation_time.observe(generation_time)
            generation_quality.set(result.quality_score)

            # Store in history for self-improvement
            if self.enable_self_improvement:
                await self._store_generation_result(specification, result)

            # Update LLM performance
            self.llm_orchestrator.update_performance(model_used, result.quality_score)

            logger.info(
                "Code generation completed",
                quality_score=result.quality_score,
                generation_time=generation_time,
                model_used=model_used,
            )

            return result

        except Exception as e:
            logger.error("Code generation failed", error=str(e))
            raise

    async def generate_from_examples(
        self,
        examples: List[Dict[str, str]],
        target_description: str,
        language: Language,
    ) -> GeneratedCode:
        """
        Generate code by learning from examples
        """
        # Extract patterns from examples
        patterns = await self._extract_patterns_from_examples(examples)

        # Create specification from patterns
        spec = await self._create_spec_from_patterns(
            patterns, target_description, language
        )

        # Generate code
        return await self.generate_code(spec)

    async def improve_existing_code(
        self, code: str, language: Language, improvement_goals: List[str]
    ) -> GeneratedCode:
        """
        Improve existing code based on specified goals
        """
        # Analyze current code
        current_metrics = await self.quality_analyzer.analyze(code, language)

        # Create improvement specification
        spec = CodeSpecification(
            name="improved_code",
            type=CodeType.MODULE,
            language=language,
            description=f"Improve code for: {', '.join(improvement_goals)}",
            requirements=improvement_goals,
            inputs={},
            outputs={},
        )

        # Generate improved version
        prompt = await self._create_improvement_prompt(
            code, improvement_goals, current_metrics
        )
        improved_code, model_used = await self.llm_orchestrator.generate(prompt)

        # Verify improvements
        new_metrics = await self.quality_analyzer.analyze(improved_code, language)

        return GeneratedCode(
            code=improved_code,
            language=language,
            type=CodeType.MODULE,
            metrics=new_metrics,
            quality_score=new_metrics.get("overall", 0),
            model_used=model_used,
            confidence=self._calculate_improvement_confidence(
                current_metrics, new_metrics
            ),
        )

    # Private methods

    def _load_pattern_library(self) -> Dict[str, CodePattern]:
        """Load reusable code patterns"""
        patterns = {}

        if self.pattern_library_path and self.pattern_library_path.exists():
            for pattern_file in self.pattern_library_path.glob("*.yaml"):
                try:
                    with open(pattern_file, "r") as f:
                        pattern_data = yaml.safe_load(f)
                        pattern = CodePattern(**pattern_data)
                        patterns[pattern.name] = pattern
                except Exception as e:
                    logger.warning(
                        f"Failed to load pattern {pattern_file}", error=str(e)
                    )

        return patterns

    async def _find_relevant_patterns(
        self, spec: CodeSpecification
    ) -> List[CodePattern]:
        """Find patterns relevant to the specification"""
        relevant = []

        for pattern in self.pattern_library.values():
            # Check language match
            if pattern.language != spec.language:
                continue

            # Check pattern relevance
            relevance_score = 0

            # Check if pattern name/description matches requirements
            for req in spec.requirements:
                if req.lower() in pattern.description.lower():
                    relevance_score += 1
                if req.lower() in pattern.name.lower():
                    relevance_score += 2

            # Check if pattern is mentioned in spec patterns
            if pattern.name in spec.patterns:
                relevance_score += 5

            # Check tag overlap
            spec_tags = set(spec.metadata.get("tags", []))
            if spec_tags & pattern.tags:
                relevance_score += len(spec_tags & pattern.tags)

            if relevance_score > 0:
                relevant.append((relevance_score, pattern))

        # Sort by relevance and return top patterns
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [pattern for _, pattern in relevant[:5]]

    async def _create_generation_prompt(
        self, spec: CodeSpecification, patterns: List[CodePattern]
    ) -> str:
        """Create detailed prompt for code generation"""
        prompt_parts = [
            f"Generate {spec.language.value} code for: {spec.name}",
            f"Type: {spec.type.value}",
            f"Description: {spec.description}",
            "",
            "Requirements:",
        ]

        for req in spec.requirements:
            prompt_parts.append(f"- {req}")

        if spec.inputs:
            prompt_parts.extend(["", "Inputs:"])
            for name, details in spec.inputs.items():
                prompt_parts.append(
                    f"- {name}: {details.get('type', 'Any')} - {details.get('description', '')}"
                )

        if spec.outputs:
            prompt_parts.extend(["", "Outputs:"])
            for name, details in spec.outputs.items():
                prompt_parts.append(
                    f"- {name}: {details.get('type', 'Any')} - {details.get('description', '')}"
                )

        if patterns:
            prompt_parts.extend(["", "Use these patterns as reference:"])
            for pattern in patterns[:3]:
                prompt_parts.append(f"\nPattern: {pattern.name}")
                prompt_parts.append(f"Template:\n{pattern.template}")

        if spec.examples:
            prompt_parts.extend(["", "Examples:"])
            for example in spec.examples[:2]:
                prompt_parts.append(f"\nInput: {example.get('input', '')}")
                prompt_parts.append(f"Output: {example.get('output', '')}")

        if spec.constraints:
            prompt_parts.extend(["", "Constraints:"])
            for key, value in spec.constraints.items():
                prompt_parts.append(f"- {key}: {value}")

        if spec.quality_requirements:
            prompt_parts.extend(["", "Quality Requirements:"])
            for metric, threshold in spec.quality_requirements.items():
                prompt_parts.append(f"- {metric}: >= {threshold}")

        prompt_parts.extend(
            [
                "",
                "Generate clean, well-documented, production-ready code.",
                "Include appropriate error handling and type hints.",
                "Follow best practices and design patterns.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _refine_code(
        self, code: str, feedback: str, spec: CodeSpecification
    ) -> str:
        """Refine code based on feedback"""
        refinement_prompt = f"""
Improve the following {spec.language.value} code based on this feedback:

Current Code:
{code}

Feedback:
{feedback}

Requirements:
- Maintain all existing functionality
- Address all feedback points
- Improve code quality
- Keep the same interface/API

Generate the improved version:
"""

        refined, _ = await self.llm_orchestrator.generate(
            refinement_prompt, strategy="best_performer"
        )

        return refined

    async def _generate_tests(self, code: str, spec: CodeSpecification) -> str:
        """Generate comprehensive tests for the code"""
        test_prompt = f"""
Generate comprehensive tests for the following {spec.language.value} code:

{code}

Requirements:
- Test all functions/methods
- Include edge cases
- Test error conditions
- Use appropriate testing framework for {spec.language.value}
- Include both unit tests and integration tests where applicable
- Add test documentation

Generate the test code:
"""

        tests, _ = await self.llm_orchestrator.generate(test_prompt)

        # Format tests
        if spec.language in self.formatters:
            tests = await self.formatters[spec.language](tests)

        return tests

    async def _generate_documentation(self, code: str, spec: CodeSpecification) -> str:
        """Generate comprehensive documentation"""
        doc_prompt = f"""
Generate comprehensive documentation for the following {spec.language.value} code:

{code}

Include:
- Overview and purpose
- Installation/setup instructions
- API documentation
- Usage examples
- Configuration options
- Troubleshooting guide
- Contributing guidelines

Format: Markdown
"""

        documentation, _ = await self.llm_orchestrator.generate(doc_prompt)
        return documentation

    async def _format_code(self, code: str, language: Language) -> str:
        """Format code according to language standards"""
        if language in self.formatters:
            return await self.formatters[language](code)
        return code

    async def _format_python(self, code: str) -> str:
        """Format Python code"""
        try:
            # Format with black
            code = black.format_str(code, mode=black.Mode())
            # Sort imports with isort
            code = isort.code(code)
            return code
        except Exception as e:
            logger.warning("Python formatting failed", error=str(e))
            return code

    async def _format_javascript(self, code: str) -> str:
        """Format JavaScript code"""
        # Would use prettier or similar
        return code

    async def _format_typescript(self, code: str) -> str:
        """Format TypeScript code"""
        # Would use prettier or similar
        return code

    async def _format_go(self, code: str) -> str:
        """Format Go code"""
        try:
            # Use gofmt
            with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
                f.write(code)
                f.flush()

                result = subprocess.run(
                    ["gofmt", f.name], capture_output=True, text=True
                )

                if result.returncode == 0:
                    return result.stdout
                else:
                    logger.warning("Go formatting failed", error=result.stderr)
                    return code
        except Exception as e:
            logger.warning("Go formatting failed", error=str(e))
            return code
        finally:
            if "f" in locals():
                os.unlink(f.name)

    async def _extract_dependencies(self, code: str, language: Language) -> List[str]:
        """Extract dependencies from code"""
        dependencies = []

        if language == Language.PYTHON:
            # Extract imports
            import_pattern = re.compile(
                r"^(?:from\s+(\S+)|import\s+(\S+))", re.MULTILINE
            )
            matches = import_pattern.findall(code)

            for match in matches:
                module = match[0] or match[1]
                # Filter out standard library modules (simplified)
                if not module.startswith(("os", "sys", "json", "math", "re")):
                    base_module = module.split(".")[0]
                    if base_module not in dependencies:
                        dependencies.append(base_module)

        elif language == Language.JAVASCRIPT or language == Language.TYPESCRIPT:
            # Extract require/import statements
            import_pattern = re.compile(
                r"(?:import.*from\s+['\"]([^'\"]+)|require\(['\"]([^'\"]+)"
            )
            matches = import_pattern.findall(code)

            for match in matches:
                module = match[0] or match[1]
                if not module.startswith(".") and module not in dependencies:
                    dependencies.append(module)

        elif language == Language.GO:
            # Extract import statements
            import_section = re.search(r"import\s*\(([^)]+)\)", code, re.DOTALL)
            if import_section:
                imports = import_section.group(1)
                for line in imports.split("\n"):
                    line = line.strip().strip('"')
                    if line and not line.startswith("//") and line not in dependencies:
                        dependencies.append(line)

        return dependencies

    def _calculate_confidence(
        self, metrics: Dict[str, float], iterations: int
    ) -> float:
        """Calculate confidence score for generated code"""
        # Base confidence on quality metrics
        quality_confidence = metrics.get("overall", 0)

        # Adjust based on iterations needed
        iteration_penalty = (iterations - 1) * 0.05

        # Factor in specific metrics
        critical_metrics = ["security", "testability"]
        critical_score = np.mean([metrics.get(m, 0) for m in critical_metrics])

        confidence = quality_confidence * 0.7 + critical_score * 0.3 - iteration_penalty

        return max(0, min(1, confidence))

    def _calculate_improvement_confidence(
        self, old_metrics: Dict[str, float], new_metrics: Dict[str, float]
    ) -> float:
        """Calculate confidence in code improvement"""
        improvements = []

        for metric, old_value in old_metrics.items():
            new_value = new_metrics.get(metric, old_value)
            if new_value > old_value:
                improvements.append((new_value - old_value) / max(old_value, 0.1))

        if not improvements:
            return 0.5

        avg_improvement = np.mean(improvements)
        return min(1, 0.5 + avg_improvement)

    async def _generate_improvement_feedback(
        self, code: str, metrics: Dict[str, float], spec: CodeSpecification
    ) -> str:
        """Generate specific improvement feedback"""
        feedback_parts = ["Improve the following aspects:"]

        # Identify weak areas
        weak_metrics = [
            (metric, score)
            for metric, score in metrics.items()
            if score < 0.7 and metric != "overall"
        ]

        # Sort by score (worst first)
        weak_metrics.sort(key=lambda x: x[1])

        for metric, score in weak_metrics[:3]:
            if metric == "readability":
                feedback_parts.append(
                    f"- Improve readability (current: {score:.2f}): "
                    "Use better variable names, add whitespace, break long lines"
                )
            elif metric == "maintainability":
                feedback_parts.append(
                    f"- Reduce complexity (current: {score:.2f}): "
                    "Break down complex functions, reduce nesting, simplify logic"
                )
            elif metric == "security":
                feedback_parts.append(
                    f"- Fix security issues (current: {score:.2f}): "
                    "Remove hardcoded secrets, validate inputs, use secure functions"
                )
            elif metric == "performance":
                feedback_parts.append(
                    f"- Optimize performance (current: {score:.2f}): "
                    "Reduce nested loops, use efficient algorithms, minimize I/O"
                )
            elif metric == "testability":
                feedback_parts.append(
                    f"- Improve testability (current: {score:.2f}): "
                    "Add dependency injection, create pure functions, reduce coupling"
                )
            elif metric == "documentation":
                feedback_parts.append(
                    f"- Add documentation (current: {score:.2f}): "
                    "Add docstrings, explain complex logic, document parameters"
                )

        # Add specific requirements not met
        if spec.quality_requirements:
            for req_metric, threshold in spec.quality_requirements.items():
                if metrics.get(req_metric, 0) < threshold:
                    feedback_parts.append(
                        f"- {req_metric} must be at least {threshold} "
                        f"(current: {metrics.get(req_metric, 0):.2f})"
                    )

        return "\n".join(feedback_parts)

    async def _store_generation_result(
        self, spec: CodeSpecification, result: GeneratedCode
    ):
        """Store generation result for self-improvement"""
        self.generation_history.append(
            {"spec": spec, "result": result, "timestamp": datetime.now()}
        )

        # Learn patterns if enabled
        if hasattr(self, "pattern_learner") and result.quality_score > 0.9:
            await self.pattern_learner.learn_from_generation(spec, result)

    async def _extract_patterns_from_examples(
        self, examples: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Extract reusable patterns from code examples"""
        patterns = []

        for example in examples:
            # Analyze structure
            structure = await self._analyze_code_structure(example.get("code", ""))

            # Extract common patterns
            if structure:
                patterns.append(
                    {
                        "structure": structure,
                        "input": example.get("input", ""),
                        "output": example.get("output", ""),
                        "patterns": structure.get("patterns", []),
                    }
                )

        return patterns

    async def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure to extract patterns"""
        structure = {"functions": [], "classes": [], "patterns": [], "complexity": 0}

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure["functions"].append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "decorators": [
                                d.id for d in node.decorator_list if hasattr(d, "id")
                            ],
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append(
                        {
                            "name": node.name,
                            "bases": [
                                base.id for base in node.bases if hasattr(base, "id")
                            ],
                            "methods": [
                                n.name
                                for n in node.body
                                if isinstance(n, ast.FunctionDef)
                            ],
                        }
                    )

            # Identify patterns
            if any("__init__" in c.get("methods", []) for c in structure["classes"]):
                structure["patterns"].append("constructor")
            if any(
                "property" in f.get("decorators", []) for f in structure["functions"]
            ):
                structure["patterns"].append("property")
            if any(
                "staticmethod" in f.get("decorators", [])
                for f in structure["functions"]
            ):
                structure["patterns"].append("staticmethod")

        except Exception as e:
            logger.warning("Failed to analyze code structure", error=str(e))

        return structure

    async def _create_spec_from_patterns(
        self, patterns: List[Dict[str, Any]], description: str, language: Language
    ) -> CodeSpecification:
        """Create specification from extracted patterns"""
        # Aggregate common elements
        all_functions = []
        all_classes = []
        all_patterns = []

        for pattern in patterns:
            all_functions.extend(pattern.get("structure", {}).get("functions", []))
            all_classes.extend(pattern.get("structure", {}).get("classes", []))
            all_patterns.extend(pattern.get("structure", {}).get("patterns", []))

        # Deduplicate
        unique_patterns = list(set(all_patterns))

        # Create specification
        return CodeSpecification(
            name="pattern_based_code",
            type=CodeType.MODULE if all_classes else CodeType.FUNCTION,
            language=language,
            description=description,
            requirements=[
                f"Implement similar structure to examples",
                f"Use patterns: {', '.join(unique_patterns)}",
            ],
            inputs={},
            outputs={},
            patterns=unique_patterns,
        )

    async def _create_improvement_prompt(
        self, code: str, goals: List[str], metrics: Dict[str, float]
    ) -> str:
        """Create prompt for code improvement"""
        prompt_parts = [
            "Improve the following code to achieve these goals:",
            "",
            "Goals:",
        ]

        for goal in goals:
            prompt_parts.append(f"- {goal}")

        prompt_parts.extend(["", "Current metrics:"])

        for metric, score in metrics.items():
            if metric != "overall":
                prompt_parts.append(f"- {metric}: {score:.2f}")

        prompt_parts.extend(
            [
                "",
                "Current code:",
                code,
                "",
                "Generate improved version that:",
                "- Maintains backward compatibility",
                "- Addresses all improvement goals",
                "- Follows best practices",
                "- Includes necessary documentation",
            ]
        )

        return "\n".join(prompt_parts)


class FeedbackLoop:
    """Manages feedback and learning from generation results"""

    def __init__(self):
        self.feedback_history = []
        self.improvement_strategies = {}

    async def process_feedback(
        self,
        spec: CodeSpecification,
        result: GeneratedCode,
        user_feedback: Optional[str] = None,
    ):
        """Process feedback to improve future generations"""
        feedback_entry = {
            "spec": spec,
            "result": result,
            "user_feedback": user_feedback,
            "timestamp": datetime.now(),
            "lessons": await self._extract_lessons(spec, result, user_feedback),
        }

        self.feedback_history.append(feedback_entry)

        # Update improvement strategies
        await self._update_strategies(feedback_entry)

    async def _extract_lessons(
        self,
        spec: CodeSpecification,
        result: GeneratedCode,
        user_feedback: Optional[str],
    ) -> List[str]:
        """Extract lessons from generation result"""
        lessons = []

        # Learn from quality metrics
        for metric, score in result.metrics.items():
            if score < 0.7:
                lessons.append(
                    f"Improve {metric} for {spec.language.value} {spec.type.value}"
                )

        # Learn from user feedback
        if user_feedback:
            if "error" in user_feedback.lower():
                lessons.append("Focus on error handling")
            if "performance" in user_feedback.lower():
                lessons.append("Prioritize performance optimization")
            if (
                "readable" in user_feedback.lower()
                or "understand" in user_feedback.lower()
            ):
                lessons.append("Improve code readability")

        return lessons

    async def _update_strategies(self, feedback_entry: Dict[str, Any]):
        """Update improvement strategies based on feedback"""
        language = feedback_entry["spec"].language
        code_type = feedback_entry["spec"].type
        key = f"{language.value}:{code_type.value}"

        if key not in self.improvement_strategies:
            self.improvement_strategies[key] = {
                "successful_patterns": [],
                "failed_patterns": [],
                "quality_trends": [],
            }

        strategy = self.improvement_strategies[key]

        # Update based on quality
        quality = feedback_entry["result"].quality_score
        strategy["quality_trends"].append(quality)

        # Track patterns
        if quality > 0.8:
            strategy["successful_patterns"].extend(feedback_entry["spec"].patterns)
        else:
            strategy["failed_patterns"].extend(feedback_entry["spec"].patterns)


class PatternLearner:
    """Learns new patterns from successful code generations"""

    def __init__(self):
        self.learned_patterns = {}
        self.pattern_performance = {}

    async def learn_from_generation(
        self, spec: CodeSpecification, result: GeneratedCode
    ):
        """Learn patterns from successful generation"""
        if result.quality_score < 0.9:
            return

        # Extract patterns from generated code
        patterns = await self._extract_patterns(result.code, spec.language)

        for pattern in patterns:
            pattern_key = f"{spec.language.value}:{pattern['type']}"

            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = []

            # Store pattern with context
            self.learned_patterns[pattern_key].append(
                {
                    "pattern": pattern,
                    "spec": spec,
                    "quality": result.quality_score,
                    "timestamp": datetime.now(),
                }
            )

            # Update performance tracking
            if pattern_key not in self.pattern_performance:
                self.pattern_performance[pattern_key] = []

            self.pattern_performance[pattern_key].append(result.quality_score)

    async def _extract_patterns(
        self, code: str, language: Language
    ) -> List[Dict[str, Any]]:
        """Extract reusable patterns from code"""
        patterns = []

        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)

                # Extract function patterns
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        pattern = {
                            "type": "function",
                            "name": node.name,
                            "decorators": [
                                d.id for d in node.decorator_list if hasattr(d, "id")
                            ],
                            "has_docstring": ast.get_docstring(node) is not None,
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                        }
                        patterns.append(pattern)

                    elif isinstance(node, ast.ClassDef):
                        pattern = {
                            "type": "class",
                            "name": node.name,
                            "bases": [
                                base.id for base in node.bases if hasattr(base, "id")
                            ],
                            "has_docstring": ast.get_docstring(node) is not None,
                            "methods": [
                                n.name
                                for n in node.body
                                if isinstance(n, ast.FunctionDef)
                            ],
                        }
                        patterns.append(pattern)

            except Exception as e:
                logger.warning("Failed to extract patterns", error=str(e))

        return patterns


# Example usage and testing
async def example_usage():
    """Example usage of the Code Generator Agent"""

    # Initialize agent
    agent = CodeGeneratorAgent()

    # Example 1: Generate a REST API wrapper
    api_spec = CodeSpecification(
        name="github_api_wrapper",
        type=CodeType.API,
        language=Language.PYTHON,
        description="A wrapper for GitHub REST API with rate limiting and caching",
        requirements=[
            "Implement rate limiting with exponential backoff",
            "Add response caching with TTL",
            "Support authentication via token",
            "Provide async and sync interfaces",
            "Handle pagination automatically",
        ],
        inputs={
            "token": {
                "type": "str",
                "description": "GitHub API token",
                "required": True,
            },
            "cache_ttl": {
                "type": "int",
                "description": "Cache TTL in seconds",
                "required": False,
            },
        },
        outputs={
            "repositories": {
                "type": "List[Dict]",
                "description": "List of repositories",
            },
            "issues": {"type": "List[Dict]", "description": "List of issues"},
        },
        patterns=["singleton", "factory"],
        quality_requirements={"testability": 0.8, "security": 0.9},
    )

    # Generate code
    result = await agent.generate_code(api_spec)

    print(f"Generated {result.language.value} code:")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Generation Time: {result.generation_time:.2f}s")
    print(f"\nCode:\n{result.code[:500]}...")

    # Example 2: Improve existing code
    existing_code = """
def process_data(data):
    result = []
    for item in data:
        if item['status'] == 'active':
            result.append(item['value'] * 2)
    return result
"""

    improved = await agent.improve_existing_code(
        existing_code,
        Language.PYTHON,
        ["Add type hints", "Improve performance", "Add error handling"],
    )

    print(f"\nImproved code:")
    print(improved.code)


if __name__ == "__main__":
    asyncio.run(example_usage())
