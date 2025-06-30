"""
Base component class for all JARVIS components
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio


class JARVISComponent(ABC):
    """
    Base class for all JARVIS components.

    Provides common functionality:
    - Initialization and shutdown
    - Status reporting
    - Health checking
    - Configuration management
    - Logging
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize JARVIS component.

        Args:
            name: Component name
            config: Component configuration
        """
        self.name = name
        self.config = config or {}
        self.initialized = False
        self.start_time = None
        self.logger = logging.getLogger(f"jarvis.{name}")
        self._status = "created"
        self._health_status = {"healthy": True, "issues": []}

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the component.
        Must be called before using the component.
        """
        self.start_time = datetime.now()
        self.initialized = True
        self._status = "initialized"
        self.logger.info(f"{self.name} initialized")

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the component.
        Should clean up resources and save state if needed.
        """
        self._status = "shutdown"
        self.initialized = False
        self.logger.info(f"{self.name} shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.

        Returns:
            Dict containing status information
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "name": self.name,
            "status": self._status,
            "initialized": self.initialized,
            "uptime_seconds": uptime,
            "health": self._health_status,
            "config": self.config,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the component.

        Returns:
            Dict with health status
        """
        # Default implementation - override in subclasses
        return self._health_status

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update component configuration.

        Args:
            config: New configuration to merge
        """
        self.config.update(config)
        self.logger.info(f"{self.name} configuration updated")

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return self._health_status.get("healthy", False)

    def _set_health_issue(self, issue: str, severity: str = "warning") -> None:
        """
        Record a health issue.

        Args:
            issue: Description of the issue
            severity: Issue severity (warning, error, critical)
        """
        self._health_status["healthy"] = False
        self._health_status["issues"].append(
            {"issue": issue, "severity": severity, "timestamp": datetime.now()}
        )

    def _clear_health_issues(self) -> None:
        """Clear all health issues"""
        self._health_status["healthy"] = True
        self._health_status["issues"] = []
