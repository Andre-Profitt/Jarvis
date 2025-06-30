"""
Test Generator for JARVIS
=========================

Advanced automated test generation system.
"""

import ast
import asyncio
import json
import re
import inspect
import typing
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Type, Callable
from pathlib import Path
import textwrap
import random
import string
from datetime import datetime
import subprocess
import tempfile
import coverage
import pytest
import hypothesis
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import faker
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge
import black
import isort
import autopep8

logger = get_logger(__name__)

# Metrics
tests_generated = Counter("tests_generated_total", "Total tests generated", ["type"])
test_generation_time = Histogram(
    "test_generation_duration_seconds", "Test generation time"
)
test_coverage_achieved = Gauge("test_coverage_percent", "Test coverage achieved")
test_quality_score = Gauge("test_quality_score", "Quality score of generated tests")


@dataclass
class TestCase:
    """Represents a generated test case"""

    name: str
    test_type: str  # unit, integration, property, fuzzing
    target_function: str
    inputs: List[Any]
    expected_output: Any
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    assertions: List[str] = field(default_factory=list)
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class TestSuite:
    """Collection of test cases for a module"""

    module_name: str
    test_cases: List[TestCase]
    imports: List[str]
    fixtures: List[str]
    setup_module: Optional[str] = None
    teardown_module: Optional[str] = None
    coverage_target: float = 80.0


@dataclass
class TestStrategy:
    """Strategy for generating test data"""

    parameter_name: str
    parameter_type: Type
    strategy: SearchStrategy
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class CoverageReport:
    """Test coverage analysis report"""

    total_coverage: float
    line_coverage: float
    branch_coverage: float
    missing_lines: List[int]
    missing_branches: List[Tuple[int, int]]
    covered_functions: Set[str]
    uncovered_functions: Set[str]


class TestGenerator:
    """
    Advanced test generation system with multiple strategies.

    Features:
    - Automatic test case generation from code analysis
    - Property-based testing with Hypothesis
    - Fuzzing test generation
    - Mock generation
    - Edge case detection
    - Coverage-guided test generation
    - Regression test generation
    - Integration test scaffolding
    """

    def __init__(
        self,
        enable_property_testing: bool = True,
        enable_fuzzing: bool = True,
        enable_mocking: bool = True,
        coverage_target: float = 80.0,
    ):

        self.enable_property_testing = enable_property_testing
        self.enable_fuzzing = enable_fuzzing
        self.enable_mocking = enable_mocking
        self.coverage_target = coverage_target

        # Test data generators
        self.faker = faker.Faker()
        self.type_strategies = self._initialize_type_strategies()

        # Analysis cache
        self.function_analysis_cache = {}
        self.coverage_data = {}

        logger.info(
            "Test Generator initialized",
            property_testing=enable_property_testing,
            fuzzing=enable_fuzzing,
            coverage_target=coverage_target,
        )

    def _initialize_type_strategies(self) -> Dict[Type, SearchStrategy]:
        """Initialize Hypothesis strategies for common types"""
        return {
            int: st.integers(),
            float: st.floats(allow_nan=False, allow_infinity=False),
            str: st.text(),
            bool: st.booleans(),
            list: st.lists(st.integers()),
            dict: st.dictionaries(st.text(), st.integers()),
            set: st.sets(st.integers()),
            tuple: st.tuples(st.integers(), st.text()),
            bytes: st.binary(),
            type(None): st.none(),
        }

    async def generate_tests_for_module(
        self, module_path: Path, output_path: Optional[Path] = None
    ) -> TestSuite:
        """Generate comprehensive tests for a Python module"""
        tests_generated.labels(type="module").inc()

        # Read and parse module
        with open(module_path, "r") as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Failed to parse {module_path}: {e}")
            return TestSuite(module_name=module_path.stem, test_cases=[], imports=[])

        # Analyze module
        module_info = self._analyze_module(tree, source_code)

        # Generate test cases
        test_cases = []

        # Generate tests for each function/method
        for func_info in module_info["functions"]:
            # Unit tests
            unit_tests = await self._generate_unit_tests(func_info)
            test_cases.extend(unit_tests)

            # Property-based tests
            if self.enable_property_testing:
                property_tests = await self._generate_property_tests(func_info)
                test_cases.extend(property_tests)

            # Fuzzing tests
            if self.enable_fuzzing:
                fuzz_tests = await self._generate_fuzz_tests(func_info)
                test_cases.extend(fuzz_tests)

        # Generate tests for classes
        for class_info in module_info["classes"]:
            class_tests = await self._generate_class_tests(class_info)
            test_cases.extend(class_tests)

        # Generate integration tests if multiple classes/functions interact
        if len(module_info["functions"]) > 1 or len(module_info["classes"]) > 0:
            integration_tests = await self._generate_integration_tests(module_info)
            test_cases.extend(integration_tests)

        # Create test suite
        test_suite = TestSuite(
            module_name=module_path.stem,
            test_cases=test_cases,
            imports=self._generate_imports(module_path, test_cases),
            fixtures=self._generate_fixtures(test_cases),
            setup_module=self._generate_setup_code(module_info),
            teardown_module=self._generate_teardown_code(module_info),
        )

        # Write test file if output path provided
        if output_path:
            test_code = self._generate_test_code(test_suite)
            with open(output_path, "w") as f:
                f.write(test_code)

            # Format the generated code
            self._format_test_code(output_path)

        # Update metrics
        tests_generated.labels(type="total").inc(len(test_cases))

        return test_suite

    def _analyze_module(self, tree: ast.AST, source_code: str) -> Dict[str, Any]:
        """Analyze module structure and extract information"""
        module_info = {
            "functions": [],
            "classes": [],
            "imports": [],
            "globals": [],
            "dependencies": set(),
        }

        class ModuleAnalyzer(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if not self._is_nested_function(node):
                    func_info = self._analyze_function(node)
                    module_info["functions"].append(func_info)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                if not self._is_nested_function(node):
                    func_info = self._analyze_function(node)
                    func_info["is_async"] = True
                    module_info["functions"].append(func_info)
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                class_info = self._analyze_class(node)
                module_info["classes"].append(class_info)
                self.generic_visit(node)

            def visit_Import(self, node):
                for alias in node.names:
                    module_info["imports"].append(
                        {"module": alias.name, "alias": alias.asname}
                    )
                    module_info["dependencies"].add(alias.name.split(".")[0])
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    module_info["dependencies"].add(node.module.split(".")[0])
                self.generic_visit(node)

            def _is_nested_function(self, node):
                # Check if function is nested inside another function
                return False  # Simplified

            def _analyze_function(self, node):
                return {
                    "name": node.name,
                    "args": self._extract_arguments(node.args),
                    "returns": self._extract_return_type(node),
                    "docstring": ast.get_docstring(node),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "complexity": self._calculate_complexity(node),
                    "calls": self._extract_function_calls(node),
                    "raises": self._extract_exceptions(node),
                    "node": node,
                }

            def _analyze_class(self, node):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(self._analyze_function(item))

                return {
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "methods": methods,
                    "docstring": ast.get_docstring(node),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "node": node,
                }

            def _extract_arguments(self, args):
                arg_info = []
                for i, arg in enumerate(args.args):
                    default = None
                    if args.defaults and i >= len(args.args) - len(args.defaults):
                        default_idx = i - (len(args.args) - len(args.defaults))
                        default = ast.unparse(args.defaults[default_idx])

                    arg_info.append(
                        {
                            "name": arg.arg,
                            "type": (
                                ast.unparse(arg.annotation) if arg.annotation else None
                            ),
                            "default": default,
                        }
                    )
                return arg_info

            def _extract_return_type(self, node):
                if node.returns:
                    return ast.unparse(node.returns)
                return None

            def _calculate_complexity(self, node):
                # Simplified cyclomatic complexity
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(
                        child, (ast.If, ast.While, ast.For, ast.ExceptHandler)
                    ):
                        complexity += 1
                return complexity

            def _extract_function_calls(self, node):
                calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            calls.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            calls.append(
                                f"{ast.unparse(child.func.value)}.{child.func.attr}"
                            )
                return calls

            def _extract_exceptions(self, node):
                exceptions = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise):
                        if child.exc:
                            if isinstance(child.exc, ast.Call) and isinstance(
                                child.exc.func, ast.Name
                            ):
                                exceptions.append(child.exc.func.id)
                            elif isinstance(child.exc, ast.Name):
                                exceptions.append(child.exc.id)
                return exceptions

        analyzer = ModuleAnalyzer()
        analyzer.visit(tree)

        return module_info

    async def _generate_unit_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Generate unit tests for a function"""
        test_cases = []

        # Generate happy path test
        happy_path = self._generate_happy_path_test(func_info)
        if happy_path:
            test_cases.append(happy_path)

        # Generate edge case tests
        edge_cases = self._generate_edge_case_tests(func_info)
        test_cases.extend(edge_cases)

        # Generate error case tests
        error_cases = self._generate_error_case_tests(func_info)
        test_cases.extend(error_cases)

        # Generate boundary value tests
        boundary_tests = self._generate_boundary_tests(func_info)
        test_cases.extend(boundary_tests)

        return test_cases

    def _generate_happy_path_test(
        self, func_info: Dict[str, Any]
    ) -> Optional[TestCase]:
        """Generate a happy path test case"""
        inputs = []

        # Generate valid inputs for each parameter
        for arg in func_info["args"]:
            if arg["name"] == "self":
                continue

            value = self._generate_valid_value(arg)
            inputs.append(value)

        # Generate test case
        test_name = f"test_{func_info['name']}_happy_path"

        # Try to infer expected output
        expected = self._infer_expected_output(func_info, inputs)

        return TestCase(
            name=test_name,
            test_type="unit",
            target_function=func_info["name"],
            inputs=inputs,
            expected_output=expected,
            assertions=(
                [
                    f"assert result is not None",
                    f"assert isinstance(result, {self._infer_return_type(func_info)})",
                ]
                if not expected
                else [f"assert result == {repr(expected)}"]
            ),
            description=f"Test {func_info['name']} with valid inputs",
            tags={"happy_path", "unit"},
        )

    def _generate_edge_case_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Generate edge case tests"""
        test_cases = []

        for i, arg in enumerate(func_info["args"]):
            if arg["name"] == "self":
                continue

            # Generate edge cases for this parameter
            edge_values = self._generate_edge_values(arg)

            for edge_value in edge_values:
                # Create inputs with edge case
                inputs = []
                for j, param in enumerate(func_info["args"]):
                    if param["name"] == "self":
                        continue

                    if j == i:
                        inputs.append(edge_value)
                    else:
                        inputs.append(self._generate_valid_value(param))

                test_name = f"test_{func_info['name']}_edge_case_{arg['name']}_{self._describe_value(edge_value)}"

                test_cases.append(
                    TestCase(
                        name=test_name,
                        test_type="unit",
                        target_function=func_info["name"],
                        inputs=inputs,
                        expected_output=None,
                        assertions=[
                            f"# Should handle edge case: {arg['name']} = {repr(edge_value)}",
                            f"assert result is not None or raises Exception",
                        ],
                        description=f"Test {func_info['name']} with edge case for {arg['name']}",
                        tags={"edge_case", "unit"},
                    )
                )

        return test_cases

    def _generate_error_case_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Generate error case tests"""
        test_cases = []

        # Test with None values
        if len(func_info["args"]) > 1:  # Skip 'self'
            none_test = TestCase(
                name=f"test_{func_info['name']}_none_input",
                test_type="unit",
                target_function=func_info["name"],
                inputs=[None] * (len(func_info["args"]) - 1),
                expected_output=None,
                assertions=[
                    f"with pytest.raises((TypeError, ValueError, AttributeError)):",
                    f"    {func_info['name']}(*inputs)",
                ],
                description=f"Test {func_info['name']} with None inputs",
                tags={"error_case", "unit"},
            )
            test_cases.append(none_test)

        # Test with wrong types
        for i, arg in enumerate(func_info["args"]):
            if arg["name"] == "self":
                continue

            wrong_type_value = self._generate_wrong_type_value(arg)
            if wrong_type_value is not None:
                inputs = []
                for j, param in enumerate(func_info["args"]):
                    if param["name"] == "self":
                        continue

                    if j == i:
                        inputs.append(wrong_type_value)
                    else:
                        inputs.append(self._generate_valid_value(param))

                test_name = f"test_{func_info['name']}_wrong_type_{arg['name']}"

                test_cases.append(
                    TestCase(
                        name=test_name,
                        test_type="unit",
                        target_function=func_info["name"],
                        inputs=inputs,
                        expected_output=None,
                        assertions=[
                            f"with pytest.raises(TypeError):",
                            f"    {func_info['name']}(*inputs)",
                        ],
                        description=f"Test {func_info['name']} with wrong type for {arg['name']}",
                        tags={"error_case", "type_error", "unit"},
                    )
                )

        return test_cases

    def _generate_boundary_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Generate boundary value tests"""
        test_cases = []

        for arg in func_info["args"]:
            if arg["name"] == "self":
                continue

            boundary_values = self._generate_boundary_values(arg)

            for boundary_value in boundary_values:
                inputs = []
                for param in func_info["args"]:
                    if param["name"] == "self":
                        continue

                    if param["name"] == arg["name"]:
                        inputs.append(boundary_value)
                    else:
                        inputs.append(self._generate_valid_value(param))

                test_name = f"test_{func_info['name']}_boundary_{arg['name']}_{self._describe_value(boundary_value)}"

                test_cases.append(
                    TestCase(
                        name=test_name,
                        test_type="unit",
                        target_function=func_info["name"],
                        inputs=inputs,
                        expected_output=None,
                        assertions=[
                            f"# Test boundary value: {arg['name']} = {repr(boundary_value)}",
                            f"result = {func_info['name']}(*inputs)",
                            f"assert result is not None",
                        ],
                        description=f"Test {func_info['name']} with boundary value for {arg['name']}",
                        tags={"boundary", "unit"},
                    )
                )

        return test_cases

    async def _generate_property_tests(
        self, func_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generate property-based tests using Hypothesis"""
        test_cases = []

        # Generate strategies for each parameter
        strategies = []
        for arg in func_info["args"]:
            if arg["name"] == "self":
                continue

            strategy = self._generate_hypothesis_strategy(arg)
            strategies.append(strategy)

        if not strategies:
            return test_cases

        # Generate property test
        test_name = f"test_{func_info['name']}_properties"

        property_checks = self._generate_property_checks(func_info)

        test_case = TestCase(
            name=test_name,
            test_type="property",
            target_function=func_info["name"],
            inputs=strategies,
            expected_output=None,
            setup_code=f"@hypothesis.given({', '.join(f'st.{s}' for s in strategies)})",
            assertions=property_checks,
            description=f"Property-based test for {func_info['name']}",
            tags={"property", "hypothesis"},
        )

        test_cases.append(test_case)

        return test_cases

    async def _generate_fuzz_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Generate fuzzing tests"""
        test_cases = []

        # Generate random inputs
        fuzz_iterations = 10

        for i in range(fuzz_iterations):
            inputs = []

            for arg in func_info["args"]:
                if arg["name"] == "self":
                    continue

                fuzz_value = self._generate_fuzz_value(arg)
                inputs.append(fuzz_value)

            test_name = f"test_{func_info['name']}_fuzz_{i}"

            test_case = TestCase(
                name=test_name,
                test_type="fuzzing",
                target_function=func_info["name"],
                inputs=inputs,
                expected_output=None,
                assertions=[
                    f"try:",
                    f"    result = {func_info['name']}(*inputs)",
                    f"    # Function should not crash",
                    f"    assert True",
                    f"except Exception as e:",
                    f"    # Log the exception for debugging",
                    f"    print(f'Fuzz test failed with: {{e}}')",
                    f"    # Re-raise if it's not an expected exception",
                    f"    if not isinstance(e, (ValueError, TypeError, KeyError)):",
                    f"        raise",
                ],
                description=f"Fuzz test #{i} for {func_info['name']}",
                tags={"fuzzing", "robustness"},
            )

            test_cases.append(test_case)

        return test_cases

    async def _generate_class_tests(self, class_info: Dict[str, Any]) -> List[TestCase]:
        """Generate tests for a class"""
        test_cases = []

        # Test class instantiation
        init_test = TestCase(
            name=f"test_{class_info['name']}_instantiation",
            test_type="unit",
            target_function=f"{class_info['name']}.__init__",
            inputs=[],
            expected_output=None,
            assertions=[
                f"instance = {class_info['name']}()",
                f"assert instance is not None",
                f"assert isinstance(instance, {class_info['name']})",
            ],
            description=f"Test {class_info['name']} instantiation",
            tags={"class", "instantiation"},
        )
        test_cases.append(init_test)

        # Test each method
        for method in class_info["methods"]:
            if method["name"] in ["__init__", "__str__", "__repr__"]:
                continue

            # Create instance setup
            setup_code = f"instance = {class_info['name']}()"

            # Generate method tests
            method_tests = await self._generate_unit_tests(method)

            # Adjust tests to use instance
            for test in method_tests:
                test.setup_code = setup_code
                test.target_function = f"instance.{method['name']}"
                test.tags.add("method")

            test_cases.extend(method_tests)

        return test_cases

    async def _generate_integration_tests(
        self, module_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generate integration tests for module components"""
        test_cases = []

        # Find functions that call other functions
        for func in module_info["functions"]:
            called_functions = [
                call
                for call in func["calls"]
                if any(f["name"] == call for f in module_info["functions"])
            ]

            if called_functions:
                test_name = f"test_{func['name']}_integration"

                test_case = TestCase(
                    name=test_name,
                    test_type="integration",
                    target_function=func["name"],
                    inputs=self._generate_integration_inputs(func),
                    expected_output=None,
                    assertions=[
                        f"# Test integration with: {', '.join(called_functions)}",
                        f"result = {func['name']}(*inputs)",
                        f"assert result is not None",
                        f"# Verify interactions occurred correctly",
                    ],
                    description=f"Integration test for {func['name']}",
                    tags={"integration"},
                )

                test_cases.append(test_case)

        return test_cases

    def _generate_valid_value(self, arg: Dict[str, Any]) -> Any:
        """Generate a valid value for a parameter"""
        param_type = arg.get("type")
        param_name = arg["name"]

        # Type-based generation
        if param_type:
            if "int" in param_type:
                return random.randint(1, 100)
            elif "float" in param_type:
                return random.uniform(0.1, 100.0)
            elif "str" in param_type:
                return self.faker.word()
            elif "bool" in param_type:
                return random.choice([True, False])
            elif "List" in param_type:
                return [random.randint(1, 10) for _ in range(3)]
            elif "Dict" in param_type:
                return {self.faker.word(): random.randint(1, 100) for _ in range(3)}

        # Name-based heuristics
        if "email" in param_name.lower():
            return self.faker.email()
        elif "name" in param_name.lower():
            return self.faker.name()
        elif "url" in param_name.lower():
            return self.faker.url()
        elif "path" in param_name.lower() or "file" in param_name.lower():
            return f"/tmp/{self.faker.file_name()}"
        elif "id" in param_name.lower():
            return random.randint(1, 1000)
        elif "count" in param_name.lower() or "size" in param_name.lower():
            return random.randint(1, 100)

        # Default
        return self.faker.word()

    def _generate_edge_values(self, arg: Dict[str, Any]) -> List[Any]:
        """Generate edge case values for a parameter"""
        edge_values = []
        param_type = arg.get("type")

        if param_type:
            if "int" in param_type:
                edge_values.extend([0, -1, 1, -(2**31), 2**31 - 1])
            elif "float" in param_type:
                edge_values.extend(
                    [0.0, -0.0, 1e-10, -1e-10, float("inf"), -float("inf")]
                )
            elif "str" in param_type:
                edge_values.extend(["", " ", "\n", "\t", "a" * 1000])
            elif "List" in param_type:
                edge_values.extend([[], [None], [[]], [1] * 1000])
            elif "Dict" in param_type:
                edge_values.extend([{}, {"": ""}, {None: None}])

        return edge_values

    def _generate_boundary_values(self, arg: Dict[str, Any]) -> List[Any]:
        """Generate boundary values for a parameter"""
        boundary_values = []
        param_type = arg.get("type")
        param_name = arg["name"]

        if param_type:
            if "int" in param_type:
                # Common boundaries
                boundary_values.extend([0, 1, -1, 10, 100, 255, 256])

                # Name-based boundaries
                if "age" in param_name.lower():
                    boundary_values.extend([0, 1, 18, 65, 120])
                elif "index" in param_name.lower():
                    boundary_values.extend([0, 1, -1])

            elif "float" in param_type:
                boundary_values.extend([0.0, 1.0, -1.0, 0.5, -0.5])

            elif "str" in param_type:
                if "length" in param_name.lower() or "size" in param_name.lower():
                    boundary_values.extend(["", "a", "a" * 255, "a" * 256])

        return boundary_values

    def _generate_wrong_type_value(self, arg: Dict[str, Any]) -> Any:
        """Generate a value of the wrong type"""
        param_type = arg.get("type")

        if param_type:
            if "int" in param_type:
                return "not_an_int"
            elif "float" in param_type:
                return "not_a_float"
            elif "str" in param_type:
                return 12345
            elif "List" in param_type:
                return "not_a_list"
            elif "Dict" in param_type:
                return ["not", "a", "dict"]
            elif "bool" in param_type:
                return "not_a_bool"

        # Default wrong type
        return object()

    def _generate_fuzz_value(self, arg: Dict[str, Any]) -> Any:
        """Generate a random fuzz value"""
        fuzz_types = [
            lambda: random.randint(-(2**32), 2**32),
            lambda: random.uniform(-1e6, 1e6),
            lambda: "".join(random.choices(string.printable, k=random.randint(0, 100))),
            lambda: [
                self._generate_fuzz_value({"type": "any"})
                for _ in range(random.randint(0, 10))
            ],
            lambda: {
                str(i): self._generate_fuzz_value({"type": "any"})
                for i in range(random.randint(0, 5))
            },
            lambda: random.choice([True, False, None]),
            lambda: bytes(
                random.randint(0, 255) for _ in range(random.randint(0, 100))
            ),
        ]

        return random.choice(fuzz_types)()

    def _generate_hypothesis_strategy(self, arg: Dict[str, Any]) -> str:
        """Generate Hypothesis strategy for a parameter"""
        param_type = arg.get("type")
        param_name = arg["name"]

        if param_type:
            if "int" in param_type:
                return "integers()"
            elif "float" in param_type:
                return "floats(allow_nan=False, allow_infinity=False)"
            elif "str" in param_type:
                return "text()"
            elif "bool" in param_type:
                return "booleans()"
            elif "List[int]" in param_type:
                return "lists(integers())"
            elif "List[str]" in param_type:
                return "lists(text())"
            elif "Dict[str, int]" in param_type:
                return "dictionaries(text(), integers())"

        # Default strategy
        return "text()"

    def _generate_property_checks(self, func_info: Dict[str, Any]) -> List[str]:
        """Generate property checks for a function"""
        checks = []

        # Basic properties
        checks.append(f"result = {func_info['name']}(*args)")

        # Return type check
        if func_info["returns"]:
            checks.append(f"assert isinstance(result, {func_info['returns']})")

        # Common properties based on function name
        func_name = func_info["name"].lower()

        if "sort" in func_name:
            checks.append(
                "assert all(result[i] <= result[i+1] for i in range(len(result)-1))"
            )
            checks.append("assert len(result) == len(input_list)")

        elif "reverse" in func_name:
            checks.append("assert result == input_data[::-1]")

        elif "add" in func_name or "sum" in func_name:
            checks.append("# Commutativity: add(a, b) == add(b, a)")
            checks.append("# Associativity: add(add(a, b), c) == add(a, add(b, c))")

        elif "multiply" in func_name:
            checks.append("# Commutativity: multiply(a, b) == multiply(b, a)")
            checks.append("# Identity: multiply(a, 1) == a")

        elif "filter" in func_name:
            checks.append("assert all(predicate(item) for item in result)")
            checks.append("assert len(result) <= len(input_data)")

        else:
            # Generic properties
            checks.append("# Function should be deterministic")
            checks.append("# Function should not modify input arguments")
            checks.append("# Function should handle edge cases gracefully")

        return checks

    def _generate_integration_inputs(self, func_info: Dict[str, Any]) -> List[Any]:
        """Generate inputs for integration testing"""
        inputs = []

        for arg in func_info["args"]:
            if arg["name"] == "self":
                continue

            # Generate realistic data for integration tests
            value = self._generate_realistic_value(arg)
            inputs.append(value)

        return inputs

    def _generate_realistic_value(self, arg: Dict[str, Any]) -> Any:
        """Generate realistic test data"""
        param_name = arg["name"].lower()

        # Domain-specific realistic data
        if "user" in param_name:
            return {
                "id": random.randint(1, 1000),
                "name": self.faker.name(),
                "email": self.faker.email(),
            }
        elif "product" in param_name:
            return {
                "id": random.randint(1, 1000),
                "name": self.faker.word(),
                "price": round(random.uniform(10, 1000), 2),
            }
        elif "order" in param_name:
            return {
                "id": random.randint(1, 10000),
                "items": [
                    {"product_id": i, "quantity": random.randint(1, 5)}
                    for i in range(random.randint(1, 5))
                ],
                "total": round(random.uniform(50, 500), 2),
            }

        # Fall back to regular generation
        return self._generate_valid_value(arg)

    def _describe_value(self, value: Any) -> str:
        """Generate a descriptive name for a value"""
        if value is None:
            return "none"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value).replace(".", "_").replace("-", "neg")
        elif isinstance(value, str):
            if not value:
                return "empty"
            elif value.isspace():
                return "whitespace"
            elif len(value) > 10:
                return "long_string"
            else:
                return value[:10].replace(" ", "_")
        elif isinstance(value, list):
            if not value:
                return "empty_list"
            else:
                return f"list_{len(value)}"
        elif isinstance(value, dict):
            if not value:
                return "empty_dict"
            else:
                return f"dict_{len(value)}"
        else:
            return type(value).__name__

    def _infer_expected_output(
        self, func_info: Dict[str, Any], inputs: List[Any]
    ) -> Any:
        """Try to infer expected output based on function analysis"""
        func_name = func_info["name"].lower()

        # Simple heuristics
        if "add" in func_name and len(inputs) == 2:
            try:
                return inputs[0] + inputs[1]
            except:
                pass

        elif "multiply" in func_name and len(inputs) == 2:
            try:
                return inputs[0] * inputs[1]
            except:
                pass

        elif "len" in func_name or "count" in func_name or "size" in func_name:
            if inputs and hasattr(inputs[0], "__len__"):
                return len(inputs[0])

        elif "is_" in func_name or func_name.startswith("is"):
            # Boolean return expected
            return True  # or False

        return None

    def _infer_return_type(self, func_info: Dict[str, Any]) -> str:
        """Infer return type from function info"""
        if func_info["returns"]:
            return func_info["returns"]

        # Heuristics based on function name
        func_name = func_info["name"].lower()

        if "is_" in func_name or func_name.startswith("is") or "check" in func_name:
            return "bool"
        elif "count" in func_name or "len" in func_name or "size" in func_name:
            return "int"
        elif "get_" in func_name or "find" in func_name:
            return "Any"
        elif "create" in func_name or "make" in func_name:
            return "object"

        return "Any"

    def _generate_imports(
        self, module_path: Path, test_cases: List[TestCase]
    ) -> List[str]:
        """Generate necessary imports for test file"""
        imports = [
            "import pytest",
            "import unittest",
            f"from {module_path.stem} import *",
        ]

        # Add imports based on test types
        test_types = {tc.test_type for tc in test_cases}

        if "property" in test_types:
            imports.extend(
                [
                    "import hypothesis",
                    "from hypothesis import strategies as st",
                    "from hypothesis import given, assume",
                ]
            )

        if "mock" in {tag for tc in test_cases for tag in tc.tags}:
            imports.append("from unittest.mock import Mock, patch, MagicMock")

        if "async" in {tag for tc in test_cases for tag in tc.tags}:
            imports.append("import asyncio")

        # Add common testing utilities
        imports.extend(
            [
                "import numpy as np",
                "import random",
                "from datetime import datetime, timedelta",
            ]
        )

        return imports

    def _generate_fixtures(self, test_cases: List[TestCase]) -> List[str]:
        """Generate pytest fixtures"""
        fixtures = []

        # Common fixtures
        fixtures.append(
            """
@pytest.fixture
def setup_data():
    '''Provide common test data'''
    return {
        'test_list': [1, 2, 3, 4, 5],
        'test_dict': {'a': 1, 'b': 2, 'c': 3},
        'test_string': 'hello world'
    }
"""
        )

        # Class instance fixtures
        class_tests = [tc for tc in test_cases if "class" in tc.tags]
        if class_tests:
            # Extract unique class names
            class_names = set()
            for tc in class_tests:
                if tc.target_function:
                    parts = tc.target_function.split(".")
                    if len(parts) > 1:
                        class_names.add(parts[0])

            for class_name in class_names:
                fixtures.append(
                    f"""
@pytest.fixture
def {class_name.lower()}_instance():
    '''Create {class_name} instance for testing'''
    return {class_name}()
"""
                )

        return fixtures

    def _generate_setup_code(self, module_info: Dict[str, Any]) -> Optional[str]:
        """Generate module setup code"""
        setup_lines = []

        # Check if module has dependencies that need setup
        if (
            "database" in module_info["dependencies"]
            or "db" in module_info["dependencies"]
        ):
            setup_lines.append("# Setup test database")
            setup_lines.append("setup_test_database()")

        if "logging" in module_info["dependencies"]:
            setup_lines.append("# Configure logging for tests")
            setup_lines.append("import logging")
            setup_lines.append("logging.basicConfig(level=logging.DEBUG)")

        if setup_lines:
            return "\n".join(setup_lines)

        return None

    def _generate_teardown_code(self, module_info: Dict[str, Any]) -> Optional[str]:
        """Generate module teardown code"""
        teardown_lines = []

        # Cleanup based on dependencies
        if (
            "database" in module_info["dependencies"]
            or "db" in module_info["dependencies"]
        ):
            teardown_lines.append("# Cleanup test database")
            teardown_lines.append("cleanup_test_database()")

        if "file" in module_info["dependencies"] or "io" in module_info["dependencies"]:
            teardown_lines.append("# Cleanup test files")
            teardown_lines.append("cleanup_test_files()")

        if teardown_lines:
            return "\n".join(teardown_lines)

        return None

    def _generate_test_code(self, test_suite: TestSuite) -> str:
        """Generate complete test file code"""
        lines = []

        # Header
        lines.append('"""')
        lines.append(f"Auto-generated tests for {test_suite.module_name}")
        lines.append(f"Generated at: {datetime.now().isoformat()}")
        lines.append('"""')
        lines.append("")

        # Imports
        for import_line in test_suite.imports:
            lines.append(import_line)
        lines.append("")

        # Module setup
        if test_suite.setup_module:
            lines.append("def setup_module():")
            lines.append("    '''Module setup'''")
            for line in test_suite.setup_module.split("\n"):
                lines.append(f"    {line}")
            lines.append("")

        # Module teardown
        if test_suite.teardown_module:
            lines.append("def teardown_module():")
            lines.append("    '''Module teardown'''")
            for line in test_suite.teardown_module.split("\n"):
                lines.append(f"    {line}")
            lines.append("")

        # Fixtures
        for fixture in test_suite.fixtures:
            lines.append(fixture.strip())
            lines.append("")

        # Group tests by class
        class_tests = {}
        function_tests = []

        for test_case in test_suite.test_cases:
            if "class" in test_case.tags and "." in test_case.target_function:
                class_name = test_case.target_function.split(".")[0]
                class_tests.setdefault(class_name, []).append(test_case)
            else:
                function_tests.append(test_case)

        # Generate test classes
        for class_name, tests in class_tests.items():
            lines.append(f"class Test{class_name}:")
            lines.append(f'    """Tests for {class_name}"""')
            lines.append("")

            for test_case in tests:
                lines.extend(self._generate_test_method(test_case, indent=1))
                lines.append("")

        # Generate test functions
        for test_case in function_tests:
            lines.extend(self._generate_test_method(test_case))
            lines.append("")

        # Additional test utilities
        lines.append("# Test execution")
        lines.append("if __name__ == '__main__':")
        lines.append("    pytest.main([__file__, '-v'])")

        return "\n".join(lines)

    def _generate_test_method(self, test_case: TestCase, indent: int = 0) -> List[str]:
        """Generate code for a single test method"""
        lines = []
        indent_str = "    " * indent

        # Function signature
        if test_case.setup_code and test_case.setup_code.startswith("@"):
            lines.append(f"{indent_str}{test_case.setup_code}")

        if test_case.test_type == "property":
            lines.append(f"{indent_str}def {test_case.name}(self, *args):")
        else:
            lines.append(f"{indent_str}def {test_case.name}(self):")

        # Docstring
        if test_case.description:
            lines.append(f'{indent_str}    """{test_case.description}"""')

        # Setup code
        if test_case.setup_code and not test_case.setup_code.startswith("@"):
            lines.append(f"{indent_str}    # Setup")
            for line in test_case.setup_code.split("\n"):
                lines.append(f"{indent_str}    {line}")

        # Test inputs
        if test_case.test_type != "property" and test_case.inputs:
            lines.append(f"{indent_str}    # Test inputs")
            lines.append(f"{indent_str}    inputs = {repr(test_case.inputs)}")

        # Assertions
        lines.append(f"{indent_str}    # Test execution and assertions")
        for assertion in test_case.assertions:
            lines.append(f"{indent_str}    {assertion}")

        # Teardown code
        if test_case.teardown_code:
            lines.append(f"{indent_str}    # Teardown")
            for line in test_case.teardown_code.split("\n"):
                lines.append(f"{indent_str}    {line}")

        return lines

    def _format_test_code(self, file_path: Path):
        """Format generated test code"""
        try:
            # Read code
            with open(file_path, "r") as f:
                code = f.read()

            # Format with black
            try:
                code = black.format_str(code, mode=black.Mode())
            except:
                # Fall back to autopep8
                code = autopep8.fix_code(code)

            # Sort imports
            code = isort.code(code)

            # Write back
            with open(file_path, "w") as f:
                f.write(code)
        except Exception as e:
            logger.warning(f"Failed to format test code: {e}")

    async def analyze_coverage(
        self, test_file: Path, source_file: Path
    ) -> CoverageReport:
        """Analyze test coverage"""
        # Run coverage analysis
        cov = coverage.Coverage()
        cov.start()

        try:
            # Run tests
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Tests failed: {result.stderr}")
        finally:
            cov.stop()
            cov.save()

        # Analyze coverage
        analysis = cov.analysis2(str(source_file))

        # Calculate metrics
        executed_lines = analysis[1]
        missing_lines = analysis[3]
        total_lines = len(executed_lines) + len(missing_lines)

        line_coverage = (
            len(executed_lines) / total_lines * 100 if total_lines > 0 else 0
        )

        # Get function coverage
        tree = ast.parse(source_file.read_text())
        all_functions = set()
        covered_functions = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                all_functions.add(node.name)
                if node.lineno not in missing_lines:
                    covered_functions.add(node.name)

        uncovered_functions = all_functions - covered_functions

        # Update metrics
        test_coverage_achieved.set(line_coverage)

        return CoverageReport(
            total_coverage=line_coverage,
            line_coverage=line_coverage,
            branch_coverage=0.0,  # Would need branch coverage plugin
            missing_lines=missing_lines,
            missing_branches=[],
            covered_functions=covered_functions,
            uncovered_functions=uncovered_functions,
        )

    async def generate_missing_tests(
        self, coverage_report: CoverageReport, source_file: Path
    ) -> TestSuite:
        """Generate tests for uncovered code"""
        # Parse source file
        with open(source_file, "r") as f:
            source_code = f.read()

        tree = ast.parse(source_code)

        # Find uncovered functions
        uncovered_funcs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in coverage_report.uncovered_functions:
                    func_info = {
                        "name": node.name,
                        "args": self._extract_arguments_from_node(node),
                        "node": node,
                    }
                    uncovered_funcs.append(func_info)

        # Generate tests for uncovered functions
        test_cases = []
        for func_info in uncovered_funcs:
            tests = await self._generate_unit_tests(func_info)
            test_cases.extend(tests)

        return TestSuite(
            module_name=source_file.stem,
            test_cases=test_cases,
            imports=self._generate_imports(source_file, test_cases),
            fixtures=[],
            coverage_target=self.coverage_target,
        )

    def _extract_arguments_from_node(
        self, node: ast.FunctionDef
    ) -> List[Dict[str, Any]]:
        """Extract argument info from AST node"""
        args = []

        for i, arg in enumerate(node.args.args):
            arg_info = {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else None,
                "default": None,
            }

            # Check for defaults
            defaults_start = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_start:
                default_idx = i - defaults_start
                arg_info["default"] = ast.unparse(node.args.defaults[default_idx])

            args.append(arg_info)

        return args


# Example usage
async def example_usage():
    """Example of using the Test Generator"""
    generator = TestGenerator()

    # Create a sample module to test
    sample_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class Calculator:
    """Simple calculator class"""
    
    def __init__(self):
        self.history = []
    
    def calculate(self, operation: str, a: float, b: float) -> float:
        """Perform calculation"""
        result = None
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        self.history.append({
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        })
        
        return result
    
    def get_history(self) -> list:
        """Get calculation history"""
        return self.history.copy()

def process_list(items: List[int]) -> List[int]:
    """Process a list of integers"""
    return [item * 2 for item in items if item > 0]
'''

    # Write sample module
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_code)
        module_path = Path(f.name)

    # Generate tests
    print("Generating tests...")
    test_suite = await generator.generate_tests_for_module(
        module_path, output_path=Path("test_generated.py")
    )

    print(f"\nGenerated {len(test_suite.test_cases)} test cases:")
    for test in test_suite.test_cases[:5]:
        print(f"  - {test.name} ({test.test_type})")

    if len(test_suite.test_cases) > 5:
        print(f"  ... and {len(test_suite.test_cases) - 5} more")

    # Show test types distribution
    test_types = {}
    for test in test_suite.test_cases:
        test_types[test.test_type] = test_types.get(test.test_type, 0) + 1

    print("\nTest type distribution:")
    for test_type, count in test_types.items():
        print(f"  {test_type}: {count}")

    # Analyze coverage
    print("\nAnalyzing coverage...")
    coverage_report = await generator.analyze_coverage(
        Path("test_generated.py"), module_path
    )

    print(f"\nCoverage Report:")
    print(f"  Total coverage: {coverage_report.total_coverage:.1f}%")
    print(f"  Covered functions: {len(coverage_report.covered_functions)}")
    print(f"  Uncovered functions: {len(coverage_report.uncovered_functions)}")

    # Clean up
    module_path.unlink()
    if Path("test_generated.py").exists():
        Path("test_generated.py").unlink()


if __name__ == "__main__":
    asyncio.run(example_usage())
