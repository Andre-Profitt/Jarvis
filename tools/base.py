"""
Base Tool Class for JARVIS Tool System
=====================================

Provides the foundation for all tools in the JARVIS ecosystem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import asyncio
import json
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools available in JARVIS"""

    WEB = "web"
    FILE = "file"
    DATA = "data"
    COMMUNICATION = "communication"
    AI = "ai"
    SYSTEM = "system"
    DEVELOPMENT = "development"
    RESEARCH = "research"
    UTILITY = "utility"


class ToolStatus(Enum):
    """Tool execution status"""

    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolMetadata:
    """Metadata for a tool"""

    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "JARVIS"
    tags: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # requests per minute
    timeout: int = 60  # seconds
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result from tool execution"""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BaseTool(ABC):
    """
    Base class for all JARVIS tools

    All tools should inherit from this class and implement the required methods.
    """

    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self.status = ToolStatus.READY
        self._rate_limiter = None
        self._execution_count = 0
        self._last_execution = None
        self._callbacks: List[Callable] = []

        # Set up rate limiting if specified
        if metadata.rate_limit:
            self._setup_rate_limiting(metadata.rate_limit)

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Internal execution method to be implemented by each tool

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool-specific result data
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate input parameters before execution

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters

        This method handles validation, rate limiting, timeouts, and error handling.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult object containing the result or error
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Update status
            self.status = ToolStatus.RUNNING
            self._last_execution = datetime.now()

            # Validate inputs
            is_valid, error_message = self.validate_inputs(**kwargs)
            if not is_valid:
                return ToolResult(
                    success=False, error=f"Validation error: {error_message}"
                )

            # Check rate limiting
            if self._rate_limiter and not await self._check_rate_limit():
                return ToolResult(
                    success=False, error="Rate limit exceeded. Please try again later."
                )

            # Execute with timeout
            try:
                result_data = await asyncio.wait_for(
                    self._execute(**kwargs), timeout=self.metadata.timeout
                )

                # Success
                self.status = ToolStatus.SUCCESS
                execution_time = asyncio.get_event_loop().time() - start_time

                result = ToolResult(
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    metadata={
                        "tool_name": self.metadata.name,
                        "execution_count": self._execution_count,
                    },
                )

                # Call success callbacks
                await self._trigger_callbacks("success", result)

                return result

            except asyncio.TimeoutError:
                self.status = ToolStatus.TIMEOUT
                return ToolResult(
                    success=False,
                    error=f"Tool execution timed out after {self.metadata.timeout} seconds",
                )

        except Exception as e:
            # Handle errors
            self.status = ToolStatus.FAILED
            error_result = ToolResult(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

            # Call error callbacks
            await self._trigger_callbacks("error", error_result)

            logger.error(f"Tool {self.metadata.name} failed: {str(e)}")
            return error_result

        finally:
            self._execution_count += 1

    def add_callback(self, callback: Callable):
        """Add a callback to be triggered after execution"""
        self._callbacks.append(callback)

    async def _trigger_callbacks(self, event_type: str, result: ToolResult):
        """Trigger all registered callbacks"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, result)
                else:
                    callback(event_type, result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _setup_rate_limiting(self, rate_limit: int):
        """Set up rate limiting for the tool"""
        # Simple rate limiting implementation
        # In production, use a proper rate limiter like aioredis-based limiter
        self._rate_limiter = {
            "rate": rate_limit,
            "window": 60,  # 1 minute window
            "requests": [],
        }

    async def _check_rate_limit(self) -> bool:
        """Check if the rate limit allows execution"""
        if not self._rate_limiter:
            return True

        now = datetime.now()
        window_start = now.timestamp() - self._rate_limiter["window"]

        # Clean old requests
        self._rate_limiter["requests"] = [
            req for req in self._rate_limiter["requests"] if req > window_start
        ]

        # Check limit
        if len(self._rate_limiter["requests"]) >= self._rate_limiter["rate"]:
            return False

        # Add current request
        self._rate_limiter["requests"].append(now.timestamp())
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current tool status"""
        return {
            "name": self.metadata.name,
            "status": self.status.value,
            "execution_count": self._execution_count,
            "last_execution": (
                self._last_execution.isoformat() if self._last_execution else None
            ),
            "rate_limit": self.metadata.rate_limit,
            "category": self.metadata.category.value,
        }

    def get_documentation(self) -> Dict[str, Any]:
        """Get tool documentation"""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "category": self.metadata.category.value,
            "version": self.metadata.version,
            "author": self.metadata.author,
            "tags": self.metadata.tags,
            "required_permissions": self.metadata.required_permissions,
            "rate_limit": self.metadata.rate_limit,
            "timeout": self.metadata.timeout,
            "examples": self.metadata.examples,
            "parameters": self._get_parameter_documentation(),
        }

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Override this method to provide parameter documentation"""
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.metadata.name}', category='{self.metadata.category.value}')"


class CompositeTool(BaseTool):
    """
    A tool that combines multiple other tools

    Useful for creating complex workflows from simpler tools.
    """

    def __init__(self, metadata: ToolMetadata, tools: List[BaseTool]):
        super().__init__(metadata)
        self.tools = tools

    async def _execute(self, **kwargs) -> Any:
        """Execute all tools in sequence"""
        results = []

        for tool in self.tools:
            result = await tool.execute(**kwargs)
            if not result.success:
                # Stop on first failure
                return {
                    "error": f"Tool {tool.metadata.name} failed: {result.error}",
                    "completed_tools": len(results),
                    "partial_results": results,
                }
            results.append(result.data)

        return {
            "success": True,
            "results": results,
            "tools_executed": [tool.metadata.name for tool in self.tools],
        }

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate inputs for all tools"""
        for tool in self.tools:
            is_valid, error = tool.validate_inputs(**kwargs)
            if not is_valid:
                return False, f"Validation failed for {tool.metadata.name}: {error}"
        return True, None


class ToolRegistry:
    """
    Registry for managing all available tools
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }

    def register(self, tool: BaseTool):
        """Register a new tool"""
        name = tool.metadata.name

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = tool
        self._categories[tool.metadata.category].append(name)

        logger.info(
            f"Registered tool: {name} (category: {tool.metadata.category.value})"
        )

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all tools or tools in a specific category"""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools"""
        return self._tools.copy()

    def search_tools(self, query: str) -> List[BaseTool]:
        """Search tools by name, description, or tags"""
        query_lower = query.lower()
        results = []

        for tool in self._tools.values():
            # Search in name
            if query_lower in tool.metadata.name.lower():
                results.append(tool)
                continue

            # Search in description
            if query_lower in tool.metadata.description.lower():
                results.append(tool)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in tool.metadata.tags):
                results.append(tool)

        return results

    def get_documentation(self) -> Dict[str, Any]:
        """Get documentation for all tools"""
        docs = {}

        for category in ToolCategory:
            tools_in_category = []
            for tool_name in self._categories[category]:
                tool = self._tools[tool_name]
                tools_in_category.append(tool.get_documentation())

            if tools_in_category:
                docs[category.value] = tools_in_category

        return docs


# Global tool registry
tool_registry = ToolRegistry()
