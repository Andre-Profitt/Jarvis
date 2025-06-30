"""
Code Improver for JARVIS
========================

Advanced code analysis and improvement system.
"""

import ast
import asyncio
import re
import tokenize
import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import subprocess
import tempfile
import difflib
import autopep8
import black
import isort
from pylint import lint
from pylint.reporters.text import TextReporter
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge
import openai
import anthropic

logger = get_logger(__name__)

# Metrics
code_improvements = Counter(
    "code_improvements_total", "Total code improvements", ["type"]
)
improvement_time = Histogram(
    "code_improvement_duration_seconds", "Code improvement time"
)
code_quality_score = Gauge(
    "code_quality_score", "Code quality metrics", ["file", "metric"]
)


@dataclass
class CodeIssue:
    """Represents a code issue or improvement opportunity"""

    file_path: str
    line_number: int
    column: int
    issue_type: str
    severity: str  # critical, major, minor, info
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    confidence: float = 1.0


@dataclass
class CodeMetrics:
    """Code quality metrics"""

    complexity: float
    maintainability_index: float
    lines_of_code: int
    comment_ratio: float
    test_coverage: Optional[float] = None
    duplication_ratio: float = 0.0
    security_score: float = 1.0
    performance_score: float = 1.0


@dataclass
class ImprovementResult:
    """Result of code improvement process"""

    original_code: str
    improved_code: str
    issues_found: List[CodeIssue]
    issues_fixed: List[CodeIssue]
    metrics_before: CodeMetrics
    metrics_after: CodeMetrics
    diff: str
    ai_suggestions: List[Dict[str, Any]] = field(default_factory=list)


class CodeImprover:
    """
    Advanced code improvement system with AI-powered suggestions.

    Features:
    - Static code analysis
    - Automatic formatting and style fixes
    - Complexity reduction suggestions
    - Performance optimization hints
    - Security vulnerability detection
    - AI-powered code review
    - Refactoring suggestions
    - Best practices enforcement
    """

    def __init__(
        self,
        enable_ai: bool = True,
        ai_providers: List[str] = ["openai", "anthropic"],
        style_guide: str = "pep8",
        max_line_length: int = 88,
        target_complexity: int = 10,
    ):

        self.enable_ai = enable_ai
        self.ai_providers = ai_providers
        self.style_guide = style_guide
        self.max_line_length = max_line_length
        self.target_complexity = target_complexity

        # Initialize AI clients if enabled
        self.ai_clients = {}
        if enable_ai:
            if "openai" in ai_providers:
                self.ai_clients["openai"] = openai.AsyncOpenAI()
            if "anthropic" in ai_providers:
                self.ai_clients["anthropic"] = anthropic.AsyncAnthropic()

        # Common code patterns to detect
        self.anti_patterns = self._load_anti_patterns()
        self.best_practices = self._load_best_practices()

        logger.info(
            "Code Improver initialized", enable_ai=enable_ai, style_guide=style_guide
        )

    def _load_anti_patterns(self) -> Dict[str, Any]:
        """Load common anti-patterns to detect"""
        return {
            "god_class": {
                "pattern": r"class\s+\w+.*?(?=class|\Z)",
                "check": lambda code: len(code.split("\n")) > 500,
                "message": "Class is too large, consider splitting into smaller classes",
            },
            "long_method": {
                "pattern": r"def\s+\w+.*?(?=def|class|\Z)",
                "check": lambda code: len(code.split("\n")) > 50,
                "message": "Method is too long, consider breaking into smaller methods",
            },
            "deep_nesting": {
                "pattern": r"(\s{16,})",
                "check": lambda match: True,
                "message": "Deep nesting detected, consider refactoring",
            },
            "magic_numbers": {
                "pattern": r"(?<!\.)\\b(?<!def )(?<!class )\d+(?!\.)",
                "check": lambda match: int(match.group()) not in [0, 1, -1],
                "message": "Magic number detected, consider using named constants",
            },
            "empty_except": {
                "pattern": r"except:\s*pass",
                "check": lambda match: True,
                "message": "Empty except block, consider handling or logging the exception",
            },
        }

    def _load_best_practices(self) -> Dict[str, Any]:
        """Load best practices to enforce"""
        return {
            "docstrings": {
                "pattern": r'(def|class)\s+\w+[^:]*:(?!\s*"{3})',
                "message": "Missing docstring",
            },
            "type_hints": {
                "pattern": r"def\s+\w+\([^)]*\)(?!\s*->)",
                "message": "Missing return type hint",
            },
            "naming_convention": {
                "pattern": r"(def|class)\s+([A-Z][a-z]+|[a-z]+[A-Z])",
                "message": "Naming convention violation",
            },
        }

    async def analyze_code(
        self, code: str, file_path: str = "<string>"
    ) -> Tuple[List[CodeIssue], CodeMetrics]:
        """Analyze code and return issues and metrics"""
        issues = []

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(
                CodeIssue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    issue_type="syntax_error",
                    severity="critical",
                    message=str(e),
                    auto_fixable=False,
                )
            )
            # Return early on syntax errors
            return issues, self._calculate_basic_metrics(code)

        # Run various analyses
        issues.extend(self._analyze_complexity(tree, code, file_path))
        issues.extend(self._analyze_style(code, file_path))
        issues.extend(self._analyze_patterns(code, file_path))
        issues.extend(self._analyze_security(tree, code, file_path))
        issues.extend(self._analyze_performance(tree, code, file_path))

        # Calculate metrics
        metrics = self._calculate_metrics(tree, code)

        # Run AI analysis if enabled
        if self.enable_ai:
            ai_issues = await self._ai_analyze(code, file_path)
            issues.extend(ai_issues)

        return issues, metrics

    def _analyze_complexity(
        self, tree: ast.AST, code: str, file_path: str
    ) -> List[CodeIssue]:
        """Analyze code complexity"""
        issues = []

        # Calculate cyclomatic complexity
        cc_results = radon_cc.cc_visit(code)

        for result in cc_results:
            if result.complexity > self.target_complexity:
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=result.lineno,
                        column=0,
                        issue_type="high_complexity",
                        severity="major" if result.complexity > 15 else "minor",
                        message=f"High cyclomatic complexity: {result.complexity}",
                        suggestion=f"Consider breaking down {result.name} into smaller functions",
                        auto_fixable=False,
                        confidence=1.0,
                    )
                )

        # Check for deeply nested code
        class NestingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
                self.deep_nodes = []

            def visit(self, node):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                    self.current_depth += 1
                    if self.current_depth > 4:
                        self.deep_nodes.append((node, self.current_depth))
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                else:
                    self.generic_visit(node)

        visitor = NestingVisitor()
        visitor.visit(tree)

        for node, depth in visitor.deep_nodes:
            issues.append(
                CodeIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type="deep_nesting",
                    severity="minor",
                    message=f"Deeply nested code (depth: {depth})",
                    suggestion="Consider extracting nested logic into separate functions",
                    auto_fixable=False,
                )
            )

        return issues

    def _analyze_style(self, code: str, file_path: str) -> List[CodeIssue]:
        """Analyze code style issues"""
        issues = []

        # Run pylint
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            # Configure pylint
            pylint_output = io.StringIO()
            reporter = TextReporter(pylint_output)

            try:
                lint.Run(
                    [f.name, "--disable=all", "--enable=C,R,W"],
                    reporter=reporter,
                    exit=False,
                )

                # Parse pylint output
                for line in pylint_output.getvalue().split("\n"):
                    if ":" in line and line[0].isdigit():
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            line_no = int(parts[0])
                            col = int(parts[1]) if parts[1].strip() else 0
                            msg_type = parts[2].strip()
                            message = parts[3].strip()

                            severity = "minor"
                            if msg_type.startswith("E"):
                                severity = "major"
                            elif msg_type.startswith("W"):
                                severity = "minor"

                            issues.append(
                                CodeIssue(
                                    file_path=file_path,
                                    line_number=line_no,
                                    column=col,
                                    issue_type="style",
                                    severity=severity,
                                    message=message,
                                    auto_fixable=msg_type
                                    in ["C0326", "C0303", "W0311"],
                                )
                            )
            except Exception as e:
                logger.warning(f"Pylint analysis failed: {e}")

            finally:
                Path(f.name).unlink()

        return issues

    def _analyze_patterns(self, code: str, file_path: str) -> List[CodeIssue]:
        """Analyze code for anti-patterns and best practices"""
        issues = []
        lines = code.split("\n")

        # Check anti-patterns
        for pattern_name, pattern_info in self.anti_patterns.items():
            for match in re.finditer(
                pattern_info["pattern"], code, re.MULTILINE | re.DOTALL
            ):
                if pattern_info["check"](
                    match.group() if hasattr(match, "group") else match
                ):
                    line_no = code[: match.start()].count("\n") + 1
                    issues.append(
                        CodeIssue(
                            file_path=file_path,
                            line_number=line_no,
                            column=match.start() - code.rfind("\n", 0, match.start()),
                            issue_type="anti_pattern",
                            severity="minor",
                            message=pattern_info["message"],
                            auto_fixable=False,
                        )
                    )

        # Check best practices
        for practice_name, practice_info in self.best_practices.items():
            for match in re.finditer(practice_info["pattern"], code, re.MULTILINE):
                line_no = code[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_no,
                        column=0,
                        issue_type="best_practice",
                        severity="info",
                        message=practice_info["message"],
                        auto_fixable=False,
                    )
                )

        return issues

    def _analyze_security(
        self, tree: ast.AST, code: str, file_path: str
    ) -> List[CodeIssue]:
        """Analyze code for security vulnerabilities"""
        issues = []

        # Check for common security issues
        security_patterns = {
            "eval_usage": {
                "pattern": r"\beval\s*\(",
                "message": "Use of eval() is a security risk",
            },
            "exec_usage": {
                "pattern": r"\bexec\s*\(",
                "message": "Use of exec() is a security risk",
            },
            "pickle_usage": {
                "pattern": r"\bpickle\.loads?\s*\(",
                "message": "Unpickling untrusted data is a security risk",
            },
            "sql_injection": {
                "pattern": r"(execute|executemany)\s*\([^)]*%[^)]*\)",
                "message": "Potential SQL injection vulnerability, use parameterized queries",
            },
            "hardcoded_password": {
                "pattern": r'(password|passwd|pwd|secret|key)\s*=\s*["\'][^"\']+["\']',
                "message": "Hardcoded password detected",
            },
        }

        for vuln_name, vuln_info in security_patterns.items():
            for match in re.finditer(vuln_info["pattern"], code, re.IGNORECASE):
                line_no = code[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_no,
                        column=match.start() - code.rfind("\n", 0, match.start()),
                        issue_type="security",
                        severity=(
                            "critical"
                            if vuln_name in ["eval_usage", "exec_usage"]
                            else "major"
                        ),
                        message=vuln_info["message"],
                        auto_fixable=False,
                        confidence=0.9,
                    )
                )

        return issues

    def _analyze_performance(
        self, tree: ast.AST, code: str, file_path: str
    ) -> List[CodeIssue]:
        """Analyze code for performance issues"""
        issues = []

        class PerformanceVisitor(ast.NodeVisitor):
            def __init__(self, issues, file_path):
                self.issues = issues
                self.file_path = file_path
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

            def visit_Call(self, node):
                # Check for inefficient operations in loops
                if self.in_loop:
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ["append", "extend"] and isinstance(
                            node.func.value, ast.Name
                        ):
                            # Repeated list operations in loop
                            self.issues.append(
                                CodeIssue(
                                    file_path=self.file_path,
                                    line_number=node.lineno,
                                    column=node.col_offset,
                                    issue_type="performance",
                                    severity="minor",
                                    message="Consider using list comprehension or preallocating list",
                                    auto_fixable=False,
                                )
                            )

                # Check for repeated file operations
                if isinstance(node.func, ast.Name) and node.func.id in [
                    "open",
                    "read",
                    "write",
                ]:
                    if self.in_loop:
                        self.issues.append(
                            CodeIssue(
                                file_path=self.file_path,
                                line_number=node.lineno,
                                column=node.col_offset,
                                issue_type="performance",
                                severity="major",
                                message="File operation inside loop, consider batching",
                                auto_fixable=False,
                            )
                        )

                self.generic_visit(node)

        visitor = PerformanceVisitor(issues, file_path)
        visitor.visit(tree)

        return issues

    async def _ai_analyze(self, code: str, file_path: str) -> List[CodeIssue]:
        """Use AI to analyze code and provide suggestions"""
        issues = []

        prompt = f"""
        Please analyze the following Python code and identify any issues, improvements, or suggestions.
        Focus on:
        1. Code quality and readability
        2. Performance optimizations
        3. Security vulnerabilities
        4. Best practices
        5. Potential bugs
        
        Code:
        ```python
        {code[:2000]}  # Truncate for API limits
        ```
        
        Provide specific line numbers and actionable suggestions.
        """

        try:
            if "openai" in self.ai_clients:
                response = await self.ai_clients["openai"].chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )

                # Parse AI response and create issues
                # This is a simplified parser - in production, use structured output
                ai_text = response.choices[0].message.content

                # Basic parsing of AI suggestions
                for line in ai_text.split("\n"):
                    if "line" in line.lower() and ":" in line:
                        # Extract line number and suggestion
                        try:
                            line_match = re.search(r"line\s*(\d+)", line, re.IGNORECASE)
                            if line_match:
                                line_no = int(line_match.group(1))
                                message = (
                                    line.split(":", 1)[1].strip()
                                    if ":" in line
                                    else line
                                )

                                issues.append(
                                    CodeIssue(
                                        file_path=file_path,
                                        line_number=line_no,
                                        column=0,
                                        issue_type="ai_suggestion",
                                        severity="info",
                                        message=message,
                                        auto_fixable=False,
                                        confidence=0.8,
                                    )
                                )
                        except:
                            pass

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")

        return issues

    def _calculate_metrics(self, tree: ast.AST, code: str) -> CodeMetrics:
        """Calculate comprehensive code metrics"""
        # Basic metrics
        lines = code.split("\n")
        loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        comment_lines = len([l for l in lines if l.strip().startswith("#")])

        # Complexity metrics
        cc_results = radon_cc.cc_visit(code)
        avg_complexity = sum(r.complexity for r in cc_results) / max(len(cc_results), 1)

        # Maintainability index
        mi = radon_metrics.mi_visit(code, True)

        # Calculate comment ratio
        comment_ratio = comment_lines / max(loc, 1)

        return CodeMetrics(
            complexity=avg_complexity,
            maintainability_index=mi,
            lines_of_code=loc,
            comment_ratio=comment_ratio,
            test_coverage=None,  # Would need external coverage data
            duplication_ratio=self._calculate_duplication(code),
            security_score=1.0,  # Simplified - would calculate based on vulnerabilities
            performance_score=1.0,  # Simplified - would calculate based on perf issues
        )

    def _calculate_basic_metrics(self, code: str) -> CodeMetrics:
        """Calculate basic metrics when full analysis isn't possible"""
        lines = code.split("\n")
        loc = len([l for l in lines if l.strip()])

        return CodeMetrics(
            complexity=0,
            maintainability_index=0,
            lines_of_code=loc,
            comment_ratio=0,
            test_coverage=None,
            duplication_ratio=0,
            security_score=0,
            performance_score=0,
        )

    def _calculate_duplication(self, code: str) -> float:
        """Calculate code duplication ratio"""
        lines = [
            l.strip()
            for l in code.split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]

        if len(lines) < 10:
            return 0.0

        # Simple duplication detection - count duplicate lines
        unique_lines = set(lines)
        duplication_ratio = 1 - (len(unique_lines) / len(lines))

        return duplication_ratio

    async def improve_code(
        self,
        code: str,
        file_path: str = "<string>",
        fix_style: bool = True,
        fix_imports: bool = True,
        fix_complexity: bool = True,
        apply_ai_suggestions: bool = True,
    ) -> ImprovementResult:
        """Improve code by fixing issues and applying suggestions"""
        original_code = code

        # Analyze original code
        issues_before, metrics_before = await self.analyze_code(code, file_path)

        # Apply automatic fixes
        improved_code = code
        fixed_issues = []

        # Fix style issues
        if fix_style:
            improved_code, style_fixed = self._fix_style(improved_code)
            fixed_issues.extend(style_fixed)

        # Fix imports
        if fix_imports:
            improved_code, import_fixed = self._fix_imports(improved_code)
            fixed_issues.extend(import_fixed)

        # Apply complexity improvements
        if fix_complexity:
            improved_code, complexity_fixed = await self._fix_complexity(improved_code)
            fixed_issues.extend(complexity_fixed)

        # Apply AI suggestions if enabled
        ai_suggestions = []
        if apply_ai_suggestions and self.enable_ai:
            improved_code, ai_suggestions = await self._apply_ai_suggestions(
                improved_code
            )

        # Analyze improved code
        issues_after, metrics_after = await self.analyze_code(improved_code, file_path)

        # Generate diff
        diff = "\n".join(
            difflib.unified_diff(
                original_code.splitlines(),
                improved_code.splitlines(),
                fromfile=f"{file_path} (original)",
                tofile=f"{file_path} (improved)",
                lineterm="",
            )
        )

        # Update metrics
        code_improvements.labels(type="total").inc()
        code_improvements.labels(type="auto_fixed").inc(len(fixed_issues))

        return ImprovementResult(
            original_code=original_code,
            improved_code=improved_code,
            issues_found=issues_before,
            issues_fixed=fixed_issues,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            diff=diff,
            ai_suggestions=ai_suggestions,
        )

    def _fix_style(self, code: str) -> Tuple[str, List[CodeIssue]]:
        """Fix style issues automatically"""
        fixed_issues = []

        try:
            # Apply autopep8
            if self.style_guide == "pep8":
                code = autopep8.fix_code(
                    code,
                    options={"max_line_length": self.max_line_length, "aggressive": 2},
                )

            # Apply black
            code = black.format_str(
                code,
                mode=black.Mode(
                    line_length=self.max_line_length, string_normalization=True
                ),
            )

            fixed_issues.append(
                CodeIssue(
                    file_path="<auto>",
                    line_number=0,
                    column=0,
                    issue_type="style",
                    severity="info",
                    message="Applied automatic style fixes",
                    auto_fixable=True,
                )
            )

        except Exception as e:
            logger.warning(f"Style fixing failed: {e}")

        return code, fixed_issues

    def _fix_imports(self, code: str) -> Tuple[str, List[CodeIssue]]:
        """Fix and sort imports"""
        fixed_issues = []

        try:
            # Apply isort
            code = isort.code(code, profile="black", line_length=self.max_line_length)

            fixed_issues.append(
                CodeIssue(
                    file_path="<auto>",
                    line_number=0,
                    column=0,
                    issue_type="imports",
                    severity="info",
                    message="Sorted and organized imports",
                    auto_fixable=True,
                )
            )

        except Exception as e:
            logger.warning(f"Import fixing failed: {e}")

        return code, fixed_issues

    async def _fix_complexity(self, code: str) -> Tuple[str, List[CodeIssue]]:
        """Apply complexity reduction transformations"""
        fixed_issues = []

        # This is a simplified version - real implementation would use AST transformations
        # to actually refactor complex code

        # Example: Extract long methods
        # This would require sophisticated AST manipulation

        return code, fixed_issues

    async def _apply_ai_suggestions(
        self, code: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply AI-generated improvements"""
        suggestions = []

        if "openai" not in self.ai_clients:
            return code, suggestions

        try:
            prompt = f"""
            Please improve the following Python code. Make it more:
            1. Readable and maintainable
            2. Performant
            3. Secure
            4. Following best practices
            
            Return ONLY the improved code, no explanations.
            
            Code:
            ```python
            {code[:2000]}
            ```
            """

            response = await self.ai_clients["openai"].chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            improved_code = response.choices[0].message.content

            # Extract code from response
            code_match = re.search(r"```python\n(.+?)```", improved_code, re.DOTALL)
            if code_match:
                improved_code = code_match.group(1)

            # Validate the improved code
            try:
                ast.parse(improved_code)
                code = improved_code
                suggestions.append(
                    {
                        "type": "ai_improvement",
                        "applied": True,
                        "description": "Applied AI-generated improvements",
                    }
                )
            except:
                suggestions.append(
                    {
                        "type": "ai_improvement",
                        "applied": False,
                        "description": "AI suggestions had syntax errors",
                    }
                )

        except Exception as e:
            logger.warning(f"AI suggestion application failed: {e}")

        return code, suggestions

    async def suggest_refactoring(
        self, code: str, file_path: str = "<string>"
    ) -> List[Dict[str, Any]]:
        """Suggest refactoring opportunities"""
        suggestions = []

        try:
            tree = ast.parse(code)

            # Analyze for refactoring opportunities
            class RefactorVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.suggestions = []
                    self.functions = {}
                    self.classes = {}

                def visit_FunctionDef(self, node):
                    # Check for duplicate functions
                    func_code = ast.unparse(node)
                    if func_code in self.functions:
                        self.suggestions.append(
                            {
                                "type": "duplicate_function",
                                "line": node.lineno,
                                "name": node.name,
                                "suggestion": f"Function '{node.name}' appears to be duplicate of '{self.functions[func_code]}'",
                            }
                        )
                    self.functions[func_code] = node.name

                    # Check for functions that could be methods
                    if len(node.args.args) > 0 and node.args.args[0].arg != "self":
                        # Check if function operates on a specific type
                        self.suggestions.append(
                            {
                                "type": "function_to_method",
                                "line": node.lineno,
                                "name": node.name,
                                "suggestion": f"Consider making '{node.name}' a method of a class",
                            }
                        )

                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    # Check for classes with too many methods
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 20:
                        self.suggestions.append(
                            {
                                "type": "large_class",
                                "line": node.lineno,
                                "name": node.name,
                                "suggestion": f"Class '{node.name}' has {len(methods)} methods, consider splitting",
                            }
                        )

                    self.generic_visit(node)

            visitor = RefactorVisitor()
            visitor.visit(tree)
            suggestions.extend(visitor.suggestions)

        except Exception as e:
            logger.error(f"Refactoring analysis failed: {e}")

        return suggestions

    async def generate_tests(self, code: str, file_path: str = "<string>") -> str:
        """Generate unit tests for the given code"""
        try:
            tree = ast.parse(code)

            # Extract functions and classes
            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node)

            # Generate test code
            test_code = "import unittest\n\n"

            for cls in classes:
                test_code += f"class Test{cls.name}(unittest.TestCase):\n"

                # Find methods in class
                methods = [
                    n
                    for n in cls.body
                    if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
                ]

                for method in methods:
                    test_code += f"    def test_{method.name}(self):\n"
                    test_code += f"        # TODO: Implement test for {method.name}\n"
                    test_code += f"        pass\n\n"

            # Generate tests for standalone functions
            if functions:
                test_code += "class TestFunctions(unittest.TestCase):\n"

                for func in functions:
                    if not any(func in ast.walk(cls) for cls in classes):
                        test_code += f"    def test_{func.name}(self):\n"
                        test_code += f"        # TODO: Implement test for {func.name}\n"
                        test_code += f"        pass\n\n"

            test_code += "\nif __name__ == '__main__':\n"
            test_code += "    unittest.main()\n"

            return test_code

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return "# Test generation failed\n"

    def get_improvement_report(self, result: ImprovementResult) -> str:
        """Generate a human-readable improvement report"""
        report = f"""
# Code Improvement Report

## Summary
- Issues Found: {len(result.issues_found)}
- Issues Fixed: {len(result.issues_fixed)}
- Lines of Code: {result.metrics_before.lines_of_code} → {result.metrics_after.lines_of_code}
- Complexity: {result.metrics_before.complexity:.2f} → {result.metrics_after.complexity:.2f}
- Maintainability: {result.metrics_before.maintainability_index:.2f} → {result.metrics_after.maintainability_index:.2f}

## Issues by Severity
"""

        # Count issues by severity
        severity_counts = {}
        for issue in result.issues_found:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        for severity in ["critical", "major", "minor", "info"]:
            if severity in severity_counts:
                report += f"- {severity.capitalize()}: {severity_counts[severity]}\n"

        report += "\n## Fixed Issues\n"
        for issue in result.issues_fixed[:10]:  # Show first 10
            report += f"- Line {issue.line_number}: {issue.message}\n"

        if len(result.issues_fixed) > 10:
            report += f"... and {len(result.issues_fixed) - 10} more\n"

        report += "\n## Remaining Issues\n"
        remaining = [i for i in result.issues_found if i not in result.issues_fixed]
        for issue in remaining[:10]:
            report += (
                f"- Line {issue.line_number} ({issue.severity}): {issue.message}\n"
            )

        if len(remaining) > 10:
            report += f"... and {len(remaining) - 10} more\n"

        if result.ai_suggestions:
            report += "\n## AI Suggestions\n"
            for suggestion in result.ai_suggestions:
                report += f"- {suggestion['description']}\n"

        return report


# Example usage
async def example_usage():
    """Example of using the Code Improver"""
    improver = CodeImprover(enable_ai=False)  # Disable AI for example

    # Sample problematic code
    bad_code = """
def calculate_total(items):
    total = 0
    for i in range(len(items)):
        total = total + items[i]
    return total

class UserManager:
    def __init__(self):
        self.users = []
    
    def add_user(self, name, password):
        # TODO: Hash password
        self.users.append({'name': name, 'password': password})
    
    def authenticate(self, name, password):
        for user in self.users:
            if user['name'] == name and user['password'] == password:
                return True
        return False
    
    def process_data(self):
        data = []
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    data.append(i * j * k)
        return data
"""

    # Analyze code
    issues, metrics = await improver.analyze_code(bad_code, "example.py")
    print(f"Found {len(issues)} issues")
    for issue in issues[:5]:
        print(f"- Line {issue.line_number}: {issue.message}")

    # Improve code
    result = await improver.improve_code(bad_code, "example.py")

    # Show improvement report
    report = improver.get_improvement_report(result)
    print("\n" + report)

    # Generate tests
    tests = await improver.generate_tests(result.improved_code)
    print("\nGenerated Tests:\n")
    print(tests[:500] + "...\n")

    # Get refactoring suggestions
    suggestions = await improver.suggest_refactoring(bad_code)
    print("\nRefactoring Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion['suggestion']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
