"""
AST Analyzer for JARVIS
=======================

Advanced Abstract Syntax Tree analysis and manipulation system.
"""

import ast
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Type
from pathlib import Path
import inspect
import textwrap
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge
import astor
import astunparse
import autopep8

logger = get_logger(__name__)

# Metrics
ast_analyses = Counter("ast_analyses_total", "Total AST analyses performed")
ast_transformations = Counter(
    "ast_transformations_total", "Total AST transformations", ["type"]
)
code_patterns_found = Counter(
    "code_patterns_found_total", "Code patterns detected", ["pattern"]
)
analysis_time = Histogram("ast_analysis_duration_seconds", "AST analysis time")


@dataclass
class CodePattern:
    """Represents a code pattern found in AST"""

    pattern_type: str
    node_type: str
    location: Tuple[int, int]  # (line, column)
    description: str
    severity: str  # info, warning, error
    context: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class FunctionSignature:
    """Represents a function signature"""

    name: str
    args: List[str]
    defaults: List[Any]
    varargs: Optional[str]
    kwargs: Optional[str]
    return_annotation: Optional[str]
    decorators: List[str]
    docstring: Optional[str]
    complexity: int = 0
    line_count: int = 0


@dataclass
class ClassInfo:
    """Represents class information"""

    name: str
    bases: List[str]
    methods: List[FunctionSignature]
    attributes: List[str]
    decorators: List[str]
    docstring: Optional[str]
    metaclass: Optional[str] = None
    is_abstract: bool = False


@dataclass
class CodeMetrics:
    """Code metrics from AST analysis"""

    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float
    halstead_metrics: Dict[str, float]
    coupling: int
    cohesion: float
    depth_of_inheritance: int
    number_of_methods: int
    number_of_attributes: int


@dataclass
class ASTDiff:
    """Represents differences between two ASTs"""

    added_nodes: List[ast.AST]
    removed_nodes: List[ast.AST]
    modified_nodes: List[Tuple[ast.AST, ast.AST]]
    moved_nodes: List[Tuple[ast.AST, ast.AST]]
    summary: str


class ASTAnalyzer:
    """
    Advanced AST analysis and manipulation system.

    Features:
    - Deep AST analysis and pattern detection
    - Code metrics calculation
    - AST transformation and refactoring
    - Control flow analysis
    - Data flow analysis
    - Type inference
    - Pattern matching
    - AST diff and merging
    """

    def __init__(self):
        self.patterns = self._load_patterns()
        self.type_inference_cache = {}
        self.call_graph = nx.DiGraph()
        self.data_flow_graph = nx.DiGraph()

        logger.info("AST Analyzer initialized")

    def _load_patterns(self) -> Dict[str, Any]:
        """Load code patterns to detect"""
        return {
            "anti_patterns": {
                "mutable_default": self._detect_mutable_default,
                "unused_variable": self._detect_unused_variable,
                "shadowed_builtin": self._detect_shadowed_builtin,
                "complex_comprehension": self._detect_complex_comprehension,
                "nested_functions": self._detect_nested_functions,
                "global_usage": self._detect_global_usage,
                "bare_except": self._detect_bare_except,
                "assert_in_production": self._detect_assert_in_production,
            },
            "code_smells": {
                "long_function": self._detect_long_function,
                "too_many_arguments": self._detect_too_many_arguments,
                "duplicate_code": self._detect_duplicate_code,
                "dead_code": self._detect_dead_code,
                "god_class": self._detect_god_class,
                "feature_envy": self._detect_feature_envy,
            },
            "security": {
                "eval_usage": self._detect_eval_usage,
                "exec_usage": self._detect_exec_usage,
                "pickle_usage": self._detect_pickle_usage,
                "sql_injection": self._detect_sql_injection,
                "hardcoded_secrets": self._detect_hardcoded_secrets,
            },
            "performance": {
                "inefficient_loop": self._detect_inefficient_loop,
                "repeated_attribute_access": self._detect_repeated_attribute_access,
                "string_concatenation_loop": self._detect_string_concatenation_loop,
                "unnecessary_list_comprehension": self._detect_unnecessary_list_comprehension,
            },
        }

    async def analyze_code(
        self, code: str, filename: str = "<string>"
    ) -> Dict[str, Any]:
        """Perform comprehensive AST analysis"""
        ast_analyses.inc()

        try:
            tree = ast.parse(code, filename)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}", "line": e.lineno, "offset": e.offset}

        # Basic analysis
        analysis = {
            "filename": filename,
            "ast": tree,
            "metrics": self._calculate_metrics(tree, code),
            "patterns": [],
            "functions": [],
            "classes": [],
            "imports": [],
            "globals": [],
            "type_annotations": {},
        }

        # Detect patterns
        for category, patterns in self.patterns.items():
            for pattern_name, detector in patterns.items():
                found_patterns = detector(tree)
                for pattern in found_patterns:
                    pattern.pattern_type = f"{category}.{pattern_name}"
                    analysis["patterns"].append(pattern)
                    code_patterns_found.labels(pattern=pattern_name).inc()

        # Extract structure
        structure_visitor = StructureVisitor()
        structure_visitor.visit(tree)

        analysis["functions"] = structure_visitor.functions
        analysis["classes"] = structure_visitor.classes
        analysis["imports"] = structure_visitor.imports
        analysis["globals"] = structure_visitor.globals

        # Type inference
        analysis["type_annotations"] = self._infer_types(tree)

        # Build call graph
        self._build_call_graph(tree)

        # Data flow analysis
        self._analyze_data_flow(tree)

        return analysis

    def _calculate_metrics(self, tree: ast.AST, code: str) -> CodeMetrics:
        """Calculate comprehensive code metrics"""
        # Line counting
        lines = code.split("\n")
        loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        # Complexity metrics
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)

        # Halstead metrics
        halstead = self._calculate_halstead_metrics(tree)

        # Coupling and cohesion
        coupling = self._calculate_coupling(tree)
        cohesion = self._calculate_cohesion(tree)

        # Calculate maintainability index
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        mi = 171
        if halstead["volume"] > 0:
            mi -= 5.2 * np.log(halstead["volume"])
        mi -= 0.23 * complexity_visitor.cyclomatic_complexity
        if loc > 0:
            mi -= 16.2 * np.log(loc)
        mi = max(0, min(100, mi))  # Normalize to 0-100

        return CodeMetrics(
            lines_of_code=loc,
            cyclomatic_complexity=complexity_visitor.cyclomatic_complexity,
            cognitive_complexity=complexity_visitor.cognitive_complexity,
            maintainability_index=mi,
            halstead_metrics=halstead,
            coupling=coupling,
            cohesion=cohesion,
            depth_of_inheritance=complexity_visitor.max_depth,
            number_of_methods=len(
                [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            ),
            number_of_attributes=len(
                [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
            ),
        )

    def _calculate_halstead_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics"""
        operator_visitor = OperatorVisitor()
        operator_visitor.visit(tree)

        n1 = len(operator_visitor.unique_operators)  # Unique operators
        n2 = len(operator_visitor.unique_operands)  # Unique operands
        N1 = operator_visitor.total_operators  # Total operators
        N2 = operator_visitor.total_operands  # Total operands

        # Halstead metrics
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * np.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        time = effort / 18  # Seconds to implement
        bugs = volume / 3000  # Estimated bugs

        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
            "time": time,
            "bugs": bugs,
        }

    def _calculate_coupling(self, tree: ast.AST) -> int:
        """Calculate coupling metric"""
        # Count external references
        coupling = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                coupling += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                coupling += len(node.names)
            elif isinstance(node, ast.Attribute):
                # Count method calls on external objects
                coupling += 1

        return coupling

    def _calculate_cohesion(self, tree: ast.AST) -> float:
        """Calculate cohesion metric (LCOM - Lack of Cohesion of Methods)"""
        class_visitor = ClassCohesionVisitor()
        class_visitor.visit(tree)

        if not class_visitor.classes:
            return 1.0  # Perfect cohesion for non-class code

        total_cohesion = 0
        for class_info in class_visitor.classes:
            # Calculate LCOM for each class
            methods = class_info["methods"]
            attributes = class_info["attributes"]

            if len(methods) <= 1 or not attributes:
                total_cohesion += 1.0
                continue

            # Count method pairs that share attributes
            shared_pairs = 0
            total_pairs = 0

            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    total_pairs += 1
                    # Check if methods share any attributes
                    attrs_i = methods[i].get("accessed_attributes", set())
                    attrs_j = methods[j].get("accessed_attributes", set())
                    if attrs_i & attrs_j:
                        shared_pairs += 1

            cohesion = shared_pairs / total_pairs if total_pairs > 0 else 1.0
            total_cohesion += cohesion

        return total_cohesion / len(class_visitor.classes)

    def _infer_types(self, tree: ast.AST) -> Dict[str, str]:
        """Infer types for variables and functions"""
        type_visitor = TypeInferenceVisitor()
        type_visitor.visit(tree)
        return type_visitor.inferred_types

    def _build_call_graph(self, tree: ast.AST):
        """Build function call graph"""
        self.call_graph.clear()

        class CallGraphVisitor(ast.NodeVisitor):
            def __init__(self, graph):
                self.graph = graph
                self.current_function = None

            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.graph.add_node(node.name)
                self.generic_visit(node)
                self.current_function = old_function

            def visit_Call(self, node):
                if self.current_function and isinstance(node.func, ast.Name):
                    self.graph.add_edge(self.current_function, node.func.id)
                self.generic_visit(node)

        visitor = CallGraphVisitor(self.call_graph)
        visitor.visit(tree)

    def _analyze_data_flow(self, tree: ast.AST):
        """Analyze data flow in the code"""
        self.data_flow_graph.clear()

        class DataFlowVisitor(ast.NodeVisitor):
            def __init__(self, graph):
                self.graph = graph
                self.definitions = {}
                self.uses = defaultdict(list)

            def visit_Assign(self, node):
                # Track variable definitions
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.definitions[target.id] = node
                        self.graph.add_node(f"def_{target.id}_{target.lineno}")
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    # Track variable uses
                    self.uses[node.id].append(node)
                    use_node = f"use_{node.id}_{node.lineno}"
                    self.graph.add_node(use_node)

                    # Connect definition to use
                    if node.id in self.definitions:
                        def_node = f"def_{node.id}_{self.definitions[node.id].lineno}"
                        self.graph.add_edge(def_node, use_node)
                self.generic_visit(node)

        visitor = DataFlowVisitor(self.data_flow_graph)
        visitor.visit(tree)

    # Pattern detection methods

    def _detect_mutable_default(self, tree: ast.AST) -> List[CodePattern]:
        """Detect mutable default arguments"""
        patterns = []

        class MutableDefaultVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                for i, default in enumerate(node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        patterns.append(
                            CodePattern(
                                pattern_type="anti_pattern",
                                node_type="FunctionDef",
                                location=(node.lineno, node.col_offset),
                                description=f"Mutable default argument in function '{node.name}'",
                                severity="warning",
                                suggestion="Use None as default and create the mutable object inside the function",
                            )
                        )
                self.generic_visit(node)

        MutableDefaultVisitor().visit(tree)
        return patterns

    def _detect_unused_variable(self, tree: ast.AST) -> List[CodePattern]:
        """Detect unused variables"""
        patterns = []

        class UnusedVariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.defined_vars = {}
                self.used_vars = set()

            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.defined_vars[target.id] = (
                            target.lineno,
                            target.col_offset,
                        )
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.used_vars.add(node.id)
                self.generic_visit(node)

            def check_unused(self):
                for var, (line, col) in self.defined_vars.items():
                    if var not in self.used_vars and not var.startswith("_"):
                        patterns.append(
                            CodePattern(
                                pattern_type="code_smell",
                                node_type="Name",
                                location=(line, col),
                                description=f"Unused variable '{var}'",
                                severity="info",
                                suggestion=f"Remove unused variable '{var}' or prefix with underscore",
                            )
                        )

        visitor = UnusedVariableVisitor()
        visitor.visit(tree)
        visitor.check_unused()
        return patterns

    def _detect_shadowed_builtin(self, tree: ast.AST) -> List[CodePattern]:
        """Detect shadowed built-in names"""
        patterns = []
        builtins = set(dir(__builtins__))

        class ShadowedBuiltinVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store) and node.id in builtins:
                    patterns.append(
                        CodePattern(
                            pattern_type="anti_pattern",
                            node_type="Name",
                            location=(node.lineno, node.col_offset),
                            description=f"Shadowing built-in name '{node.id}'",
                            severity="warning",
                            suggestion=f"Use a different name instead of '{node.id}'",
                        )
                    )
                self.generic_visit(node)

        ShadowedBuiltinVisitor().visit(tree)
        return patterns

    def _detect_complex_comprehension(self, tree: ast.AST) -> List[CodePattern]:
        """Detect overly complex comprehensions"""
        patterns = []

        class ComplexComprehensionVisitor(ast.NodeVisitor):
            def check_complexity(self, node, comp_type):
                # Count nested comprehensions and conditions
                nested_count = len(
                    [
                        n
                        for n in ast.walk(node)
                        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp))
                    ]
                )
                if_count = len([n for n in ast.walk(node) if isinstance(n, ast.If)])

                if nested_count > 1 or if_count > 2:
                    patterns.append(
                        CodePattern(
                            pattern_type="code_smell",
                            node_type=comp_type,
                            location=(node.lineno, node.col_offset),
                            description=f"Complex {comp_type} with {nested_count} nested comprehensions and {if_count} conditions",
                            severity="warning",
                            suggestion="Consider breaking this into multiple steps or using a regular loop",
                        )
                    )

            def visit_ListComp(self, node):
                self.check_complexity(node, "ListComp")
                self.generic_visit(node)

            def visit_SetComp(self, node):
                self.check_complexity(node, "SetComp")
                self.generic_visit(node)

            def visit_DictComp(self, node):
                self.check_complexity(node, "DictComp")
                self.generic_visit(node)

        ComplexComprehensionVisitor().visit(tree)
        return patterns

    def _detect_nested_functions(self, tree: ast.AST) -> List[CodePattern]:
        """Detect deeply nested functions"""
        patterns = []

        class NestedFunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0

            def visit_FunctionDef(self, node):
                self.depth += 1
                if self.depth > 2:
                    patterns.append(
                        CodePattern(
                            pattern_type="code_smell",
                            node_type="FunctionDef",
                            location=(node.lineno, node.col_offset),
                            description=f"Function '{node.name}' is nested {self.depth} levels deep",
                            severity="warning",
                            suggestion="Consider refactoring to reduce nesting",
                        )
                    )
                self.generic_visit(node)
                self.depth -= 1

        NestedFunctionVisitor().visit(tree)
        return patterns

    def _detect_global_usage(self, tree: ast.AST) -> List[CodePattern]:
        """Detect global variable usage"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                patterns.append(
                    CodePattern(
                        pattern_type="anti_pattern",
                        node_type="Global",
                        location=(node.lineno, node.col_offset),
                        description=f"Use of global variables: {', '.join(node.names)}",
                        severity="warning",
                        suggestion="Consider using function parameters or class attributes instead",
                    )
                )

        return patterns

    def _detect_bare_except(self, tree: ast.AST) -> List[CodePattern]:
        """Detect bare except clauses"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                patterns.append(
                    CodePattern(
                        pattern_type="anti_pattern",
                        node_type="ExceptHandler",
                        location=(node.lineno, node.col_offset),
                        description="Bare except clause catches all exceptions",
                        severity="warning",
                        suggestion="Specify the exception type(s) to catch",
                    )
                )

        return patterns

    def _detect_assert_in_production(self, tree: ast.AST) -> List[CodePattern]:
        """Detect assert statements that might be disabled in production"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                patterns.append(
                    CodePattern(
                        pattern_type="anti_pattern",
                        node_type="Assert",
                        location=(node.lineno, node.col_offset),
                        description="Assert statement may be disabled with -O flag",
                        severity="info",
                        suggestion="Use explicit validation with exceptions for production code",
                    )
                )

        return patterns

    def _detect_long_function(self, tree: ast.AST) -> List[CodePattern]:
        """Detect functions that are too long"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                if hasattr(node, "end_lineno"):
                    lines = node.end_lineno - node.lineno
                    if lines > 50:
                        patterns.append(
                            CodePattern(
                                pattern_type="code_smell",
                                node_type="FunctionDef",
                                location=(node.lineno, node.col_offset),
                                description=f"Function '{node.name}' is {lines} lines long",
                                severity="warning",
                                suggestion="Consider breaking this function into smaller functions",
                            )
                        )

        return patterns

    def _detect_too_many_arguments(self, tree: ast.AST) -> List[CodePattern]:
        """Detect functions with too many arguments"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arg_count = len(node.args.args) + len(node.args.kwonlyargs)
                if arg_count > 5:
                    patterns.append(
                        CodePattern(
                            pattern_type="code_smell",
                            node_type="FunctionDef",
                            location=(node.lineno, node.col_offset),
                            description=f"Function '{node.name}' has {arg_count} arguments",
                            severity="warning",
                            suggestion="Consider using a configuration object or builder pattern",
                        )
                    )

        return patterns

    def _detect_duplicate_code(self, tree: ast.AST) -> List[CodePattern]:
        """Detect duplicate code blocks"""
        patterns = []

        # Extract all function bodies
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Normalize function body for comparison
                body_str = astunparse.unparse(node.body)
                if body_str in functions:
                    patterns.append(
                        CodePattern(
                            pattern_type="code_smell",
                            node_type="FunctionDef",
                            location=(node.lineno, node.col_offset),
                            description=f"Function '{node.name}' has duplicate code with '{functions[body_str]}'",
                            severity="warning",
                            suggestion="Extract common code into a shared function",
                        )
                    )
                else:
                    functions[body_str] = node.name

        return patterns

    def _detect_dead_code(self, tree: ast.AST) -> List[CodePattern]:
        """Detect unreachable code"""
        patterns = []

        class DeadCodeVisitor(ast.NodeVisitor):
            def check_after_terminal(self, stmts, start_idx):
                for i in range(start_idx + 1, len(stmts)):
                    patterns.append(
                        CodePattern(
                            pattern_type="code_smell",
                            node_type=type(stmts[i]).__name__,
                            location=(stmts[i].lineno, stmts[i].col_offset),
                            description="Unreachable code detected",
                            severity="warning",
                            suggestion="Remove unreachable code",
                        )
                    )

            def visit_FunctionDef(self, node):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        self.check_after_terminal(node.body, i)
                        break
                self.generic_visit(node)

        DeadCodeVisitor().visit(tree)
        return patterns

    def _detect_god_class(self, tree: ast.AST) -> List[CodePattern]:
        """Detect classes that do too much"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    patterns.append(
                        CodePattern(
                            pattern_type="code_smell",
                            node_type="ClassDef",
                            location=(node.lineno, node.col_offset),
                            description=f"Class '{node.name}' has {len(methods)} methods",
                            severity="warning",
                            suggestion="Consider splitting this class based on responsibilities",
                        )
                    )

        return patterns

    def _detect_feature_envy(self, tree: ast.AST) -> List[CodePattern]:
        """Detect methods that use another class more than their own"""
        patterns = []

        # This is a simplified detection - real feature envy detection is complex
        class FeatureEnvyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None
                self.current_method = None
                self.attribute_access = defaultdict(int)

            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node):
                if self.current_class:
                    old_method = self.current_method
                    self.current_method = node.name
                    self.attribute_access.clear()
                    self.generic_visit(node)

                    # Check if method accesses external attributes more than self
                    self_access = self.attribute_access.get("self", 0)
                    max_other = max(
                        (v for k, v in self.attribute_access.items() if k != "self"),
                        default=0,
                    )

                    if max_other > self_access and self_access < 3:
                        patterns.append(
                            CodePattern(
                                pattern_type="code_smell",
                                node_type="FunctionDef",
                                location=(node.lineno, node.col_offset),
                                description=f"Method '{node.name}' may have feature envy",
                                severity="info",
                                suggestion="Consider moving this method to the class it uses most",
                            )
                        )

                    self.current_method = old_method

            def visit_Attribute(self, node):
                if self.current_method and isinstance(node.value, ast.Name):
                    self.attribute_access[node.value.id] += 1
                self.generic_visit(node)

        FeatureEnvyVisitor().visit(tree)
        return patterns

    def _detect_eval_usage(self, tree: ast.AST) -> List[CodePattern]:
        """Detect eval usage"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "eval":
                    patterns.append(
                        CodePattern(
                            pattern_type="security",
                            node_type="Call",
                            location=(node.lineno, node.col_offset),
                            description="Use of eval() is a security risk",
                            severity="error",
                            suggestion="Use ast.literal_eval() for safe evaluation or refactor to avoid eval",
                        )
                    )

        return patterns

    def _detect_exec_usage(self, tree: ast.AST) -> List[CodePattern]:
        """Detect exec usage"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "exec":
                    patterns.append(
                        CodePattern(
                            pattern_type="security",
                            node_type="Call",
                            location=(node.lineno, node.col_offset),
                            description="Use of exec() is a security risk",
                            severity="error",
                            suggestion="Refactor code to avoid dynamic execution",
                        )
                    )

        return patterns

    def _detect_pickle_usage(self, tree: ast.AST) -> List[CodePattern]:
        """Detect pickle usage on untrusted data"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ["loads", "load"] and isinstance(
                    node.func.value, ast.Name
                ):
                    if node.func.value.id == "pickle":
                        patterns.append(
                            CodePattern(
                                pattern_type="security",
                                node_type="Call",
                                location=(node.lineno, node.col_offset),
                                description="Unpickling data can execute arbitrary code",
                                severity="warning",
                                suggestion="Use JSON or other safe serialization formats for untrusted data",
                            )
                        )

        return patterns

    def _detect_sql_injection(self, tree: ast.AST) -> List[CodePattern]:
        """Detect potential SQL injection vulnerabilities"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ["execute", "executemany"]:
                    # Check if using string formatting
                    if node.args and isinstance(
                        node.args[0], (ast.BinOp, ast.JoinedStr)
                    ):
                        patterns.append(
                            CodePattern(
                                pattern_type="security",
                                node_type="Call",
                                location=(node.lineno, node.col_offset),
                                description="Potential SQL injection vulnerability",
                                severity="error",
                                suggestion="Use parameterized queries instead of string formatting",
                            )
                        )

        return patterns

    def _detect_hardcoded_secrets(self, tree: ast.AST) -> List[CodePattern]:
        """Detect hardcoded secrets"""
        patterns = []
        secret_patterns = ["password", "secret", "key", "token", "api_key"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(pattern in var_name for pattern in secret_patterns):
                            if isinstance(node.value, ast.Constant) and isinstance(
                                node.value.value, str
                            ):
                                if len(node.value.value) > 5:  # Likely a real secret
                                    patterns.append(
                                        CodePattern(
                                            pattern_type="security",
                                            node_type="Assign",
                                            location=(node.lineno, node.col_offset),
                                            description=f"Hardcoded secret in variable '{target.id}'",
                                            severity="error",
                                            suggestion="Use environment variables or secure key management",
                                        )
                                    )

        return patterns

    def _detect_inefficient_loop(self, tree: ast.AST) -> List[CodePattern]:
        """Detect inefficient loop patterns"""
        patterns = []

        class InefficientLoopVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                # Check for range(len()) pattern
                if isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name
                ):
                    if node.iter.func.id == "range" and node.iter.args:
                        if isinstance(node.iter.args[0], ast.Call):
                            if (
                                isinstance(node.iter.args[0].func, ast.Name)
                                and node.iter.args[0].func.id == "len"
                            ):
                                patterns.append(
                                    CodePattern(
                                        pattern_type="performance",
                                        node_type="For",
                                        location=(node.lineno, node.col_offset),
                                        description="Using range(len()) is inefficient",
                                        severity="info",
                                        suggestion="Use enumerate() or iterate directly over the sequence",
                                    )
                                )
                self.generic_visit(node)

        InefficientLoopVisitor().visit(tree)
        return patterns

    def _detect_repeated_attribute_access(self, tree: ast.AST) -> List[CodePattern]:
        """Detect repeated attribute access in loops"""
        patterns = []

        class RepeatedAccessVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_loop = False
                self.attribute_access = defaultdict(int)

            def visit_For(self, node):
                old_in_loop = self.in_loop
                self.in_loop = True
                self.attribute_access.clear()
                self.generic_visit(node)

                # Check for repeated access
                for attr, count in self.attribute_access.items():
                    if count > 3:
                        patterns.append(
                            CodePattern(
                                pattern_type="performance",
                                node_type="For",
                                location=(node.lineno, node.col_offset),
                                description=f"Attribute '{attr}' accessed {count} times in loop",
                                severity="info",
                                suggestion="Cache the attribute value in a local variable",
                            )
                        )

                self.in_loop = old_in_loop

            def visit_Attribute(self, node):
                if self.in_loop:
                    attr_chain = []
                    current = node
                    while isinstance(current, ast.Attribute):
                        attr_chain.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        attr_chain.append(current.id)

                    full_attr = ".".join(reversed(attr_chain))
                    self.attribute_access[full_attr] += 1

                self.generic_visit(node)

        RepeatedAccessVisitor().visit(tree)
        return patterns

    def _detect_string_concatenation_loop(self, tree: ast.AST) -> List[CodePattern]:
        """Detect string concatenation in loops"""
        patterns = []

        class StringConcatVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_loop = False

            def visit_For(self, node):
                old_in_loop = self.in_loop
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = old_in_loop

            def visit_While(self, node):
                old_in_loop = self.in_loop
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = old_in_loop

            def visit_AugAssign(self, node):
                if self.in_loop and isinstance(node.op, ast.Add):
                    if isinstance(node.target, ast.Name):
                        # Check if it's likely a string
                        patterns.append(
                            CodePattern(
                                pattern_type="performance",
                                node_type="AugAssign",
                                location=(node.lineno, node.col_offset),
                                description="String concatenation in loop is inefficient",
                                severity="warning",
                                suggestion="Use list.append() and ''.join() instead",
                            )
                        )
                self.generic_visit(node)

        StringConcatVisitor().visit(tree)
        return patterns

    def _detect_unnecessary_list_comprehension(
        self, tree: ast.AST
    ) -> List[CodePattern]:
        """Detect list comprehensions that should be generator expressions"""
        patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if passing list comp to functions that accept iterables
                if isinstance(node.func, ast.Name) and node.func.id in [
                    "sum",
                    "any",
                    "all",
                    "min",
                    "max",
                ]:
                    if node.args and isinstance(node.args[0], ast.ListComp):
                        patterns.append(
                            CodePattern(
                                pattern_type="performance",
                                node_type="ListComp",
                                location=(node.args[0].lineno, node.args[0].col_offset),
                                description=f"Unnecessary list comprehension passed to {node.func.id}()",
                                severity="info",
                                suggestion="Use a generator expression instead (remove square brackets)",
                            )
                        )

        return patterns

    def transform_ast(self, tree: ast.AST, transformations: List[str]) -> ast.AST:
        """Apply transformations to AST"""
        transformed = tree

        for transformation in transformations:
            if transformation == "remove_dead_code":
                transformer = DeadCodeRemover()
                transformed = transformer.visit(transformed)
            elif transformation == "optimize_comprehensions":
                transformer = ComprehensionOptimizer()
                transformed = transformer.visit(transformed)
            elif transformation == "fix_mutable_defaults":
                transformer = MutableDefaultFixer()
                transformed = transformer.visit(transformed)

            ast_transformations.labels(type=transformation).inc()

        return transformed

    def ast_to_code(self, tree: ast.AST, format: bool = True) -> str:
        """Convert AST back to code"""
        code = astunparse.unparse(tree)

        if format:
            code = autopep8.fix_code(code)

        return code

    def compare_asts(self, tree1: ast.AST, tree2: ast.AST) -> ASTDiff:
        """Compare two ASTs and return differences"""
        diff = ASTDiff(
            added_nodes=[],
            removed_nodes=[],
            modified_nodes=[],
            moved_nodes=[],
            summary="",
        )

        # Create node mappings
        nodes1 = {self._node_signature(node): node for node in ast.walk(tree1)}
        nodes2 = {self._node_signature(node): node for node in ast.walk(tree2)}

        # Find differences
        sig1 = set(nodes1.keys())
        sig2 = set(nodes2.keys())

        # Removed nodes
        for sig in sig1 - sig2:
            diff.removed_nodes.append(nodes1[sig])

        # Added nodes
        for sig in sig2 - sig1:
            diff.added_nodes.append(nodes2[sig])

        # Modified nodes (simplified check)
        for sig in sig1 & sig2:
            if astunparse.unparse(nodes1[sig]) != astunparse.unparse(nodes2[sig]):
                diff.modified_nodes.append((nodes1[sig], nodes2[sig]))

        # Generate summary
        diff.summary = (
            f"Added: {len(diff.added_nodes)}, "
            f"Removed: {len(diff.removed_nodes)}, "
            f"Modified: {len(diff.modified_nodes)}"
        )

        return diff

    def _node_signature(self, node: ast.AST) -> str:
        """Generate a signature for an AST node"""
        sig_parts = [type(node).__name__]

        if hasattr(node, "name"):
            sig_parts.append(f"name={node.name}")
        if hasattr(node, "id"):
            sig_parts.append(f"id={node.id}")
        if hasattr(node, "lineno"):
            sig_parts.append(f"line={node.lineno}")

        return ":".join(sig_parts)

    def visualize_ast(self, tree: ast.AST, output_path: Path = None):
        """Visualize AST as a graph"""
        graph = nx.DiGraph()

        def add_nodes(node, parent=None):
            node_id = id(node)
            label = type(node).__name__

            if hasattr(node, "name"):
                label += f"\n{node.name}"
            elif hasattr(node, "id"):
                label += f"\n{node.id}"

            graph.add_node(node_id, label=label)

            if parent:
                graph.add_edge(parent, node_id)

            for child in ast.iter_child_nodes(node):
                add_nodes(child, node_id)

        add_nodes(tree)

        # Draw graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        labels = nx.get_node_attributes(graph, "label")
        nx.draw(
            graph,
            pos,
            labels=labels,
            node_size=3000,
            node_color="lightblue",
            font_size=8,
            arrows=True,
        )

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


# Visitor classes


class StructureVisitor(ast.NodeVisitor):
    """Extract code structure information"""

    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.globals = []
        self.current_class = None

    def visit_FunctionDef(self, node):
        sig = FunctionSignature(
            name=node.name,
            args=[arg.arg for arg in node.args.args],
            defaults=[ast.unparse(d) for d in node.args.defaults],
            varargs=node.args.vararg.arg if node.args.vararg else None,
            kwargs=node.args.kwarg.arg if node.args.kwarg else None,
            return_annotation=ast.unparse(node.returns) if node.returns else None,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
            line_count=(
                node.end_lineno - node.lineno if hasattr(node, "end_lineno") else 0
            ),
        )

        if self.current_class:
            self.current_class.methods.append(sig)
        else:
            self.functions.append(sig)

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        old_class = self.current_class

        self.current_class = ClassInfo(
            name=node.name,
            bases=[ast.unparse(base) for base in node.bases],
            methods=[],
            attributes=[],
            decorators=[ast.unparse(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
        )

        self.generic_visit(node)

        self.classes.append(self.current_class)
        self.current_class = old_class

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(
                {"module": alias.name, "alias": alias.asname, "type": "import"}
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(
                {
                    "module": (
                        f"{node.module}.{alias.name}" if node.module else alias.name
                    ),
                    "alias": alias.asname,
                    "type": "from",
                    "level": node.level,
                }
            )
        self.generic_visit(node)

    def visit_Global(self, node):
        self.globals.extend(node.names)
        self.generic_visit(node)


class ComplexityVisitor(ast.NodeVisitor):
    """Calculate code complexity metrics"""

    def __init__(self):
        self.cyclomatic_complexity = 1
        self.cognitive_complexity = 0
        self.nesting_level = 0
        self.max_depth = 0

    def visit_If(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self._increase_nesting()
        self.generic_visit(node)
        self._decrease_nesting()

    def visit_While(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self._increase_nesting()
        self.generic_visit(node)
        self._decrease_nesting()

    def visit_For(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self._increase_nesting()
        self.generic_visit(node)
        self._decrease_nesting()

    def visit_ExceptHandler(self, node):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Each additional boolean operator adds complexity
        self.cyclomatic_complexity += len(node.values) - 1
        self.cognitive_complexity += len(node.values) - 1
        self.generic_visit(node)

    def _increase_nesting(self):
        self.nesting_level += 1
        self.max_depth = max(self.max_depth, self.nesting_level)

    def _decrease_nesting(self):
        self.nesting_level -= 1


class OperatorVisitor(ast.NodeVisitor):
    """Count operators and operands for Halstead metrics"""

    def __init__(self):
        self.operators = []
        self.operands = []

    @property
    def unique_operators(self):
        return set(self.operators)

    @property
    def unique_operands(self):
        return set(self.operands)

    @property
    def total_operators(self):
        return len(self.operators)

    @property
    def total_operands(self):
        return len(self.operands)

    def visit_BinOp(self, node):
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self.operators.append(type(op).__name__)
        self.generic_visit(node)

    def visit_Call(self, node):
        self.operators.append("Call")
        if isinstance(node.func, ast.Name):
            self.operands.append(node.func.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store)):
            self.operands.append(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node):
        self.operands.append(str(node.value))
        self.generic_visit(node)


class TypeInferenceVisitor(ast.NodeVisitor):
    """Infer types from code"""

    def __init__(self):
        self.inferred_types = {}
        self.context = {}

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.inferred_types[node.target.id] = ast.unparse(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Infer from type annotations
        for arg in node.args.args:
            if arg.annotation:
                self.inferred_types[f"{node.name}.{arg.arg}"] = ast.unparse(
                    arg.annotation
                )

        if node.returns:
            self.inferred_types[f"{node.name}.return"] = ast.unparse(node.returns)

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Simple type inference from literals
        for target in node.targets:
            if isinstance(target, ast.Name):
                if isinstance(node.value, ast.Constant):
                    self.inferred_types[target.id] = type(node.value.value).__name__
                elif isinstance(node.value, ast.List):
                    self.inferred_types[target.id] = "list"
                elif isinstance(node.value, ast.Dict):
                    self.inferred_types[target.id] = "dict"
                elif isinstance(node.value, ast.Set):
                    self.inferred_types[target.id] = "set"

        self.generic_visit(node)


class ClassCohesionVisitor(ast.NodeVisitor):
    """Analyze class cohesion"""

    def __init__(self):
        self.classes = []
        self.current_class = None
        self.current_method = None

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = {"name": node.name, "methods": [], "attributes": set()}

        self.generic_visit(node)

        self.classes.append(self.current_class)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        if self.current_class:
            old_method = self.current_method
            self.current_method = {"name": node.name, "accessed_attributes": set()}

            self.generic_visit(node)

            self.current_class["methods"].append(self.current_method)
            self.current_method = old_method
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node):
        if self.current_method and isinstance(node.value, ast.Name):
            if node.value.id == "self":
                self.current_method["accessed_attributes"].add(node.attr)
                self.current_class["attributes"].add(node.attr)
        self.generic_visit(node)


# AST Transformers


class DeadCodeRemover(ast.NodeTransformer):
    """Remove dead code from AST"""

    def visit_FunctionDef(self, node):
        new_body = []
        for i, stmt in enumerate(node.body):
            new_body.append(self.visit(stmt))
            if isinstance(stmt, (ast.Return, ast.Raise)):
                break  # Everything after is dead code
        node.body = new_body
        return node


class ComprehensionOptimizer(ast.NodeTransformer):
    """Optimize comprehensions"""

    def visit_Call(self, node):
        # Convert list comp to generator exp for certain functions
        if isinstance(node.func, ast.Name) and node.func.id in [
            "sum",
            "any",
            "all",
            "min",
            "max",
        ]:
            if node.args and isinstance(node.args[0], ast.ListComp):
                # Convert to generator expression
                gen_exp = ast.GeneratorExp(
                    elt=node.args[0].elt, generators=node.args[0].generators
                )
                node.args[0] = gen_exp
        return self.generic_visit(node)


class MutableDefaultFixer(ast.NodeTransformer):
    """Fix mutable default arguments"""

    def visit_FunctionDef(self, node):
        for i, default in enumerate(node.args.defaults):
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                # Replace with None
                node.args.defaults[i] = ast.Constant(value=None)

                # Add initialization in function body
                arg_idx = len(node.args.args) - len(node.args.defaults) + i
                arg_name = node.args.args[arg_idx].arg

                # Create if statement: if arg is None: arg = []
                init_stmt = ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=arg_name, ctx=ast.Load()),
                        ops=[ast.Is()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id=arg_name, ctx=ast.Store())],
                            value=default,
                        )
                    ],
                    orelse=[],
                )

                # Insert at beginning of function body
                node.body.insert(0, init_stmt)

        return self.generic_visit(node)


# Example usage
async def example_usage():
    """Example of using the AST Analyzer"""
    analyzer = ASTAnalyzer()

    # Sample code to analyze
    code = """
def calculate_total(items=[]):  # Mutable default
    total = 0
    for i in range(len(items)):  # Inefficient loop
        total += items[i]
    return total

class DataProcessor:
    def __init__(self):
        self.data = []
        
    def process(self, items):
        result = []
        for item in items:
            if item > 0:
                result.append(item * 2)
        return result
        
    def analyze(self):
        # Complex list comprehension
        processed = [[x * y for x in row if x > 0] for row in self.data if len(row) > 0]
        return sum([len(row) for row in processed])  # Unnecessary list comp

def unsafe_eval(user_input):
    return eval(user_input)  # Security risk

password = "secret123"  # Hardcoded secret
"""

    # Analyze code
    print("Analyzing code...")
    analysis = await analyzer.analyze_code(code, "example.py")

    # Print metrics
    print(f"\nCode Metrics:")
    metrics = analysis["metrics"]
    print(f"  Lines of Code: {metrics.lines_of_code}")
    print(f"  Cyclomatic Complexity: {metrics.cyclomatic_complexity}")
    print(f"  Maintainability Index: {metrics.maintainability_index:.2f}")
    print(f"  Halstead Volume: {metrics.halstead_metrics['volume']:.2f}")

    # Print detected patterns
    print(f"\nDetected Patterns ({len(analysis['patterns'])} found):")
    for pattern in analysis["patterns"]:
        print(
            f"  [{pattern.severity}] Line {pattern.location[0]}: {pattern.description}"
        )
        if pattern.suggestion:
            print(f"    → {pattern.suggestion}")

    # Print structure
    print(f"\nCode Structure:")
    print(f"  Functions: {[f.name for f in analysis['functions']]}")
    print(f"  Classes: {[c.name for c in analysis['classes']]}")

    # Transform AST
    print("\nApplying transformations...")
    tree = analysis["ast"]
    transformed = analyzer.transform_ast(
        tree, ["fix_mutable_defaults", "optimize_comprehensions"]
    )

    # Convert back to code
    improved_code = analyzer.ast_to_code(transformed)
    print("\nImproved code preview:")
    print(improved_code[:500] + "..." if len(improved_code) > 500 else improved_code)


if __name__ == "__main__":
    asyncio.run(example_usage())
