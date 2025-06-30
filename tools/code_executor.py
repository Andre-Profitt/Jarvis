"""
Code Executor Tool for JARVIS
=============================

Provides safe code execution capabilities with sandboxing and resource limits.
"""

import asyncio
import subprocess
import tempfile
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import resource
import signal
import ast
import black
import isort
from contextlib import contextmanager
import docker
import time

from .base import BaseTool, ToolMetadata, ToolCategory


class CodeExecutorTool(BaseTool):
    """
    Safe code execution tool with multiple execution environments

    Features:
    - Multiple language support (Python, JavaScript, Shell, etc.)
    - Sandboxed execution with resource limits
    - Docker container execution for isolation
    - Code formatting and linting
    - Output capture and error handling
    - Execution history tracking
    """

    def __init__(
        self, enable_docker: bool = False, docker_image: str = "python:3.9-slim"
    ):
        metadata = ToolMetadata(
            name="code_executor",
            description="Execute code safely with sandboxing and resource limits",
            category=ToolCategory.DEVELOPMENT,
            version="1.0.0",
            tags=["code", "execution", "development", "sandbox"],
            required_permissions=["code_execution"],
            rate_limit=30,  # 30 executions per minute
            timeout=30,
            examples=[
                {
                    "description": "Execute Python code",
                    "params": {
                        "code": "print('Hello, JARVIS!')\nresult = 2 + 2\nprint(f'Result: {result}')",
                        "language": "python",
                    },
                },
                {
                    "description": "Execute with input data",
                    "params": {
                        "code": "data = input_data['numbers']\nresult = sum(data)\nprint(f'Sum: {result}')",
                        "language": "python",
                        "input_data": {"numbers": [1, 2, 3, 4, 5]},
                    },
                },
            ],
        )
        super().__init__(metadata)

        self.enable_docker = enable_docker
        self.docker_image = docker_image
        self.docker_client = None

        if enable_docker:
            try:
                self.docker_client = docker.from_env()
                # Pull image if not available
                self._ensure_docker_image()
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
                self.enable_docker = False

        # Supported languages and their executors
        self.languages = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "shell": self._execute_shell,
            "bash": self._execute_shell,
            "sql": self._execute_sql,
        }

        # Execution history
        self.execution_history = []

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate code execution parameters"""
        code = kwargs.get("code")

        if not code:
            return False, "Code parameter is required"

        if not isinstance(code, str):
            return False, "Code must be a string"

        if len(code) > 100000:  # 100KB limit
            return False, "Code too large (max 100KB)"

        language = kwargs.get("language", "python")
        if language not in self.languages:
            return False, f"Unsupported language: {language}"

        # Validate code syntax for Python
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"

        # Check for dangerous patterns
        if self._contains_dangerous_code(code, language):
            return False, "Code contains potentially dangerous operations"

        return True, None

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute code with selected method"""
        code = kwargs.get("code")
        language = kwargs.get("language", "python")
        input_data = kwargs.get("input_data", {})
        use_docker = kwargs.get("use_docker", self.enable_docker)
        format_code = kwargs.get("format_code", False)

        # Format code if requested
        if format_code and language == "python":
            try:
                code = black.format_str(code, mode=black.Mode())
                code = isort.code(code)
            except:
                pass  # Continue with unformatted code

        # Execute with appropriate method
        if use_docker and self.docker_client:
            result = await self._execute_in_docker(code, language, input_data)
        else:
            executor = self.languages.get(language)
            if not executor:
                raise ValueError(f"No executor for language: {language}")
            result = await executor(code, input_data)

        # Store in history
        self.execution_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "language": language,
                "code_length": len(code),
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0),
            }
        )

        # Limit history size
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

        return result

    async def _execute_python(
        self, code: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment"""
        start_time = time.time()

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Prepare code with input data
            setup_code = f"""
import sys
import json

# Inject input data
input_data = {json.dumps(input_data)}

# Capture output
_output_buffer = []
_original_print = print

def print(*args, **kwargs):
    output = ' '.join(str(arg) for arg in args)
    _output_buffer.append(output)
    _original_print(*args, **kwargs)

# User code
try:
    {code}
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{str(e)}}")
    import traceback
    traceback.print_exc()

# Output results
_result = {{
    'output': _output_buffer,
    'variables': {{k: v for k, v in locals().items() 
                   if not k.startswith('_') and k not in ['sys', 'json', 'input_data']}}
}}
print('__RESULT_START__')
print(json.dumps(_result))
print('__RESULT_END__')
"""
            f.write(setup_code)
            f.flush()
            temp_file = f.name

        try:
            # Execute with resource limits
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=(
                    self._set_resource_limits if sys.platform != "win32" else None
                ),
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.metadata.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": "Execution timed out",
                    "execution_time": time.time() - start_time,
                }

            # Parse output
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")

            # Extract result
            result_data = {"output": [], "variables": {}}
            if "__RESULT_START__" in stdout_str and "__RESULT_END__" in stdout_str:
                start_idx = stdout_str.find("__RESULT_START__") + len(
                    "__RESULT_START__"
                )
                end_idx = stdout_str.find("__RESULT_END__")
                result_json = stdout_str[start_idx:end_idx].strip()
                try:
                    result_data = json.loads(result_json)
                except:
                    pass

            # Extract regular output
            regular_output = stdout_str
            if "__RESULT_START__" in regular_output:
                regular_output = regular_output[
                    : regular_output.find("__RESULT_START__")
                ]

            return {
                "success": process.returncode == 0,
                "output": result_data.get("output", []),
                "stdout": regular_output.strip(),
                "stderr": stderr_str.strip(),
                "variables": result_data.get("variables", {}),
                "return_code": process.returncode,
                "execution_time": time.time() - start_time,
                "language": "python",
            }

        finally:
            # Clean up
            os.unlink(temp_file)

    async def _execute_javascript(
        self, code: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js"""
        start_time = time.time()

        # Check if Node.js is available
        try:
            node_version = subprocess.run(
                ["node", "--version"], capture_output=True, text=True
            )
            if node_version.returncode != 0:
                return {
                    "success": False,
                    "error": "Node.js is not installed",
                    "execution_time": 0,
                }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Node.js is not installed",
                "execution_time": 0,
            }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            setup_code = f"""
// Inject input data
const inputData = {json.dumps(input_data)};

// Capture console output
const _outputs = [];
const _originalLog = console.log;
console.log = (...args) => {{
    _outputs.push(args.join(' '));
    _originalLog(...args);
}};

// User code
try {{
    {code}
}} catch (error) {{
    console.log(`Error: ${{error.name}}: ${{error.message}}`);
    console.log(error.stack);
}}

// Output results
console.log('__RESULT_START__');
console.log(JSON.stringify({{
    output: _outputs
}}));
console.log('__RESULT_END__');
"""
            f.write(setup_code)
            f.flush()
            temp_file = f.name

        try:
            # Execute
            process = await asyncio.create_subprocess_exec(
                "node",
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse output
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")

            # Extract result
            result_data = {"output": []}
            if "__RESULT_START__" in stdout_str and "__RESULT_END__" in stdout_str:
                start_idx = stdout_str.find("__RESULT_START__") + len(
                    "__RESULT_START__"
                )
                end_idx = stdout_str.find("__RESULT_END__")
                result_json = stdout_str[start_idx:end_idx].strip()
                try:
                    result_data = json.loads(result_json)
                except:
                    pass

            return {
                "success": process.returncode == 0,
                "output": result_data.get("output", []),
                "stdout": stdout_str.strip(),
                "stderr": stderr_str.strip(),
                "return_code": process.returncode,
                "execution_time": time.time() - start_time,
                "language": "javascript",
            }

        finally:
            os.unlink(temp_file)

    async def _execute_shell(
        self, code: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute shell commands safely"""
        start_time = time.time()

        # Basic safety check
        dangerous_commands = ["rm -rf", "dd if=", "mkfs", "format", ":(){ :|:& };:"]
        for cmd in dangerous_commands:
            if cmd in code:
                return {
                    "success": False,
                    "error": f"Dangerous command detected: {cmd}",
                    "execution_time": 0,
                }

        try:
            # Execute in shell
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=(
                    self._set_resource_limits if sys.platform != "win32" else None
                ),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.metadata.timeout
            )

            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8").splitlines(),
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
                "return_code": process.returncode,
                "execution_time": time.time() - start_time,
                "language": "shell",
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "error": "Execution timed out",
                "execution_time": time.time() - start_time,
            }

    async def _execute_sql(
        self, code: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute SQL queries (mock implementation)"""
        # This is a mock implementation
        # In production, connect to actual database
        return {
            "success": True,
            "output": ["SQL execution not implemented in demo"],
            "stdout": "SQL queries would be executed here",
            "stderr": "",
            "return_code": 0,
            "execution_time": 0.001,
            "language": "sql",
            "note": "This is a mock implementation",
        }

    async def _execute_in_docker(
        self, code: str, language: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code in Docker container for better isolation"""
        if not self.docker_client:
            return {
                "success": False,
                "error": "Docker not available",
                "execution_time": 0,
            }

        start_time = time.time()

        try:
            # Create container
            container = self.docker_client.containers.run(
                self.docker_image,
                command=f"{language} -c '{code}'",
                detach=True,
                mem_limit="512m",
                cpu_quota=50000,  # 50% CPU
                remove=False,
            )

            # Wait for completion
            exit_code = container.wait(timeout=self.metadata.timeout)

            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

            # Remove container
            container.remove()

            return {
                "success": exit_code["StatusCode"] == 0,
                "output": stdout.splitlines(),
                "stdout": stdout,
                "stderr": stderr,
                "return_code": exit_code["StatusCode"],
                "execution_time": time.time() - start_time,
                "language": language,
                "environment": "docker",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def _set_resource_limits(self):
        """Set resource limits for subprocess (Unix only)"""
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))

        # Memory limit (512MB)
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        # Limit number of processes
        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))

    def _contains_dangerous_code(self, code: str, language: str) -> bool:
        """Check for potentially dangerous code patterns"""
        dangerous_patterns = {
            "python": [
                "__import__('os')",
                "exec(",
                "eval(",
                "compile(",
                "__builtins__",
                "open('/etc/passwd'",
                "subprocess",
                "socket",
            ],
            "javascript": [
                "require('child_process')",
                "require('fs')",
                "eval(",
                "Function(",
                "process.exit",
            ],
            "shell": [
                "rm -rf /",
                ":(){ :|:& };:",  # Fork bomb
                "dd if=/dev/zero",
                "mkfs",
                "> /dev/sda",
            ],
        }

        patterns = dangerous_patterns.get(language, [])
        code_lower = code.lower()

        for pattern in patterns:
            if pattern.lower() in code_lower:
                return True

        return False

    def _ensure_docker_image(self):
        """Ensure Docker image is available"""
        try:
            self.docker_client.images.get(self.docker_image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self.docker_image}")
            self.docker_client.images.pull(self.docker_image)

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Document the parameters for this tool"""
        return {
            "code": {
                "type": "string",
                "description": "The code to execute",
                "required": True,
                "max_length": 100000,
            },
            "language": {
                "type": "string",
                "description": "Programming language",
                "required": False,
                "default": "python",
                "enum": list(self.languages.keys()),
            },
            "input_data": {
                "type": "object",
                "description": "Input data available to the code",
                "required": False,
                "default": {},
            },
            "use_docker": {
                "type": "boolean",
                "description": "Execute in Docker container for better isolation",
                "required": False,
                "default": False,
            },
            "format_code": {
                "type": "boolean",
                "description": "Format code before execution (Python only)",
                "required": False,
                "default": False,
            },
        }


# Example usage
async def example_usage():
    """Example of using the CodeExecutorTool"""
    tool = CodeExecutorTool(enable_docker=False)

    # Execute Python code
    result = await tool.execute(
        code="""
import math

# Calculate fibonacci
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate first 10 fibonacci numbers
fib_numbers = [fibonacci(i) for i in range(10)]
print(f"Fibonacci numbers: {fib_numbers}")

# Use input data
if 'numbers' in input_data:
    total = sum(input_data['numbers'])
    average = total / len(input_data['numbers'])
    print(f"Sum: {total}, Average: {average:.2f}")
""",
        language="python",
        input_data={"numbers": [10, 20, 30, 40, 50]},
    )

    if result.success:
        print("Execution successful!")
        print("Output:", "\n".join(result.data["output"]))
        print("Variables:", result.data.get("variables", {}))
    else:
        print(f"Execution failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(example_usage())
