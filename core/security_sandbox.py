#!/usr/bin/env python3
"""
Security Sandbox for JARVIS
Safe code execution environment with resource limits and isolation
"""

import os
import sys
import subprocess
import tempfile
import shutil
import resource
import signal
import time
import ast
import builtins
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
import docker
import asyncio
import psutil
from contextlib import contextmanager
import structlog

from .config_manager import config_manager
from .monitoring import monitor_performance, monitoring_service

logger = structlog.get_logger()


class SecurityViolation(Exception):
    """Raised when code violates security policies"""

    pass


class ResourceLimitExceeded(Exception):
    """Raised when code exceeds resource limits"""

    pass


class SafeBuiltins:
    """Safe subset of Python builtins"""

    ALLOWED_BUILTINS = {
        # Basic types
        "bool",
        "int",
        "float",
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "bytes",
        "bytearray",
        # Functions
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "chr",
        "divmod",
        "enumerate",
        "filter",
        "format",
        "hex",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "reversed",
        "round",
        "sorted",
        "sum",
        "zip",
        # Exceptions
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        # Constants
        "True",
        "False",
        "None",
        # Safe operations
        "print",  # Will be redirected
    }

    @classmethod
    def get_safe_builtins(cls) -> Dict[str, Any]:
        """Get dictionary of safe builtins"""
        return {
            name: getattr(builtins, name)
            for name in cls.ALLOWED_BUILTINS
            if hasattr(builtins, name)
        }


class CodeValidator:
    """Validates code for security issues"""

    FORBIDDEN_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "shutil",
        "pathlib",
        "__builtins__",
        "eval",
        "exec",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
    }

    FORBIDDEN_ATTRIBUTES = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__code__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__module__",
        "__dict__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "mro",
    }

    def __init__(self):
        self.allowed_imports = set(
            config_manager.get(
                "security.allowed_imports",
                ["math", "datetime", "itertools", "collections", "json", "re"],
            )
        )

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code for security issues"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for forbidden constructs
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    modules = [node.module] if node.module else []

                for module in modules:
                    if module and module.split(".")[0] in self.FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import: {module}"
                    if module and module.split(".")[0] not in self.allowed_imports:
                        return False, f"Import not allowed: {module}"

            # Check for eval/exec
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile", "__import__"]:
                        return False, f"Forbidden function: {node.func.id}"

            # Check for attribute access to forbidden attributes
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    return False, f"Forbidden attribute: {node.attr}"

        return True, None


class ResourceMonitor:
    """Monitors resource usage during execution"""

    def __init__(self, max_memory_mb: int = 512, max_cpu_percent: int = 50):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.process = None
        self.start_time = None
        self.monitoring = False

    def start_monitoring(self, pid: int):
        """Start monitoring a process"""
        try:
            self.process = psutil.Process(pid)
            self.start_time = time.time()
            self.monitoring = True
        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} not found")

    def check_limits(self) -> Tuple[bool, Optional[str]]:
        """Check if resource limits are exceeded"""
        if not self.monitoring or not self.process:
            return True, None

        try:
            # Check memory
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                return (
                    False,
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb}MB",
                )

            # Check CPU
            cpu_percent = self.process.cpu_percent(interval=0.1)
            if cpu_percent > self.max_cpu_percent:
                return (
                    False,
                    f"CPU limit exceeded: {cpu_percent:.1f}% > {self.max_cpu_percent}%",
                )

            return True, None

        except psutil.NoSuchProcess:
            return True, None  # Process ended

    def get_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        if not self.monitoring or not self.process:
            return {}

        try:
            return {
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                "cpu_percent": self.process.cpu_percent(interval=0.1),
                "runtime_seconds": time.time() - self.start_time,
            }
        except psutil.NoSuchProcess:
            return {}


class SecuritySandbox:
    """Main security sandbox for code execution"""

    def __init__(self):
        self.validator = CodeValidator()
        self.config = config_manager.get_synthesis_config()
        self.docker_available = self._check_docker()

        # Resource limits
        self.timeout = config_manager.get("security.sandbox_timeout", 10)
        self.max_memory = self._parse_memory_limit(
            config_manager.get("security.sandbox_memory_limit", "512MB")
        )

        logger.info(f"Security sandbox initialized (Docker: {self.docker_available})")

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def _parse_memory_limit(self, limit: str) -> int:
        """Parse memory limit string to bytes"""
        limit = limit.upper()
        if limit.endswith("GB"):
            return int(limit[:-2]) * 1024 * 1024 * 1024
        elif limit.endswith("MB"):
            return int(limit[:-2]) * 1024 * 1024
        elif limit.endswith("KB"):
            return int(limit[:-2]) * 1024
        else:
            return int(limit)

    @monitor_performance("security_sandbox")
    async def execute_code(
        self,
        code: str,
        test_inputs: Optional[Dict[str, Any]] = None,
        use_docker: bool = True,
    ) -> Dict[str, Any]:
        """Execute code in sandbox"""

        # Validate code first
        valid, error = self.validator.validate(code)
        if not valid:
            raise SecurityViolation(error)

        # Choose execution method
        if use_docker and self.docker_available:
            return await self._execute_docker(code, test_inputs)
        else:
            return await self._execute_subprocess(code, test_inputs)

    async def _execute_docker(
        self, code: str, test_inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute code in Docker container"""
        client = docker.from_env()

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = Path(tmpdir) / "code.py"

            # Prepare execution script
            exec_script = self._prepare_execution_script(code, test_inputs)
            code_file.write_text(exec_script)

            # Run in container
            try:
                container = client.containers.run(
                    "python:3.9-slim",
                    f"python /code/code.py",
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    mem_limit=self.max_memory,
                    cpu_percent=50,
                    network_disabled=True,
                    remove=True,
                    stdout=True,
                    stderr=True,
                    timeout=self.timeout,
                )

                output = container.decode("utf-8")

                # Parse output
                return self._parse_output(output)

            except docker.errors.ContainerError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": e.stderr.decode("utf-8") if e.stderr else "",
                }
            except Exception as e:
                return {"success": False, "error": f"Docker execution failed: {str(e)}"}

    async def _execute_subprocess(
        self, code: str, test_inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute code in subprocess with restrictions"""

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "code.py"

            # Prepare execution script
            exec_script = self._prepare_execution_script(code, test_inputs)
            code_file.write_text(exec_script)

            # Set resource limits for subprocess
            def set_limits():
                # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
                # Memory limit
                resource.setrlimit(
                    resource.RLIMIT_AS, (self.max_memory, self.max_memory)
                )
                # Disable core dumps
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                # Limit number of processes
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

            # Run subprocess
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    preexec_fn=set_limits if os.name != "nt" else None,
                    cwd=tmpdir,
                )

                # Monitor resources
                monitor = ResourceMonitor(
                    max_memory_mb=self.max_memory // (1024 * 1024), max_cpu_percent=50
                )
                monitor.start_monitoring(process.pid)

                # Wait for completion with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=self.timeout
                    )

                    # Check resource usage
                    ok, limit_error = monitor.check_limits()
                    if not ok:
                        return {
                            "success": False,
                            "error": limit_error,
                            "resource_usage": monitor.get_usage(),
                        }

                    # Parse output
                    output = stdout.decode("utf-8")
                    error_output = stderr.decode("utf-8")

                    if process.returncode != 0:
                        return {
                            "success": False,
                            "error": error_output or "Execution failed",
                            "output": output,
                            "resource_usage": monitor.get_usage(),
                        }

                    result = self._parse_output(output)
                    result["resource_usage"] = monitor.get_usage()
                    return result

                except asyncio.TimeoutError:
                    process.kill()
                    return {
                        "success": False,
                        "error": f"Execution timeout ({self.timeout}s)",
                        "resource_usage": monitor.get_usage(),
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Subprocess execution failed: {str(e)}",
                }

    def _prepare_execution_script(
        self, code: str, test_inputs: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare code for execution with safety wrapper"""

        safe_builtins = SafeBuiltins.get_safe_builtins()

        # Import allowed modules
        imports = "\n".join(
            [f"import {module}" for module in self.validator.allowed_imports]
        )

        # Prepare test inputs
        if test_inputs:
            input_setup = "\n".join(
                [f"{name} = {repr(value)}" for name, value in test_inputs.items()]
            )
        else:
            input_setup = ""

        # Create execution template
        template = f"""
{imports}
import json
import sys

# Redirect output
output = []
def safe_print(*args, **kwargs):
    output.append(' '.join(str(arg) for arg in args))

# Set safe builtins
__builtins__ = {safe_builtins}
__builtins__['print'] = safe_print

# Test inputs
{input_setup}

# User code
try:
    {code}
    
    # Collect results
    results = {{
        'success': True,
        'output': output,
        'variables': {{k: v for k, v in locals().items() 
                      if not k.startswith('_') and k not in ['output', 'safe_print', 'json', 'sys']}}
    }}
    
except Exception as e:
    results = {{
        'success': False,
        'error': str(e),
        'output': output
    }}

# Output results as JSON
print(json.dumps(results))
"""

        return template

    def _parse_output(self, output: str) -> Dict[str, Any]:
        """Parse execution output"""
        try:
            # Try to parse as JSON
            lines = output.strip().split("\n")
            if lines:
                # Look for JSON in last line
                for line in reversed(lines):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

            # Fallback to raw output
            return {"success": True, "output": output.split("\n"), "variables": {}}

        except Exception as e:
            return {
                "success": False,
                "error": f"Output parsing failed: {str(e)}",
                "raw_output": output,
            }

    async def test_code(
        self, code: str, test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Test code with multiple test cases"""
        results = []

        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i + 1}/{len(test_cases)}")

            try:
                result = await self.execute_code(code, test_case.get("inputs"))

                # Check expected output if provided
                if "expected" in test_case and "variables" in result:
                    for var, expected in test_case["expected"].items():
                        actual = result["variables"].get(var)
                        result["test_passed"] = actual == expected
                        if not result["test_passed"]:
                            result["test_error"] = (
                                f"Expected {var}={expected}, got {actual}"
                            )
                            break

                results.append(result)

            except Exception as e:
                results.append({"success": False, "error": str(e), "test_case": i + 1})

        return results


# Global sandbox instance
security_sandbox = SecuritySandbox()


# Convenience functions
async def safe_execute(code: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute code safely in sandbox"""
    return await security_sandbox.execute_code(code, inputs)


async def validate_code(code: str) -> Tuple[bool, Optional[str]]:
    """Validate code for security issues"""
    return security_sandbox.validator.validate(code)
