"""
Base integration class for external services
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from .component import JARVISComponent


class JARVISIntegration(JARVISComponent):
    """
    Base class for external service integrations.

    Extends JARVISComponent with integration-specific functionality:
    - Connection management
    - Authentication
    - Rate limiting
    - Error handling
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize integration.

        Args:
            name: Integration name
            config: Integration configuration
        """
        super().__init__(name, config)
        self.connected = False
        self.connection = None
        self.rate_limiter = None

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to external service.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from external service.
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test if connection is working.

        Returns:
            True if connection is healthy
        """
        pass

    async def initialize(self) -> None:
        """Initialize and connect to service"""
        await super().initialize()

        try:
            self.connected = await self.connect()
            if self.connected:
                self._status = "connected"
                self.logger.info(f"{self.name} connected successfully")
            else:
                self._status = "connection_failed"
                self._set_health_issue("Failed to connect", "error")
                self.logger.error(f"{self.name} connection failed")
        except Exception as e:
            self._status = "error"
            self._set_health_issue(f"Connection error: {str(e)}", "critical")
            self.logger.error(f"{self.name} initialization error: {e}")
            raise

    async def shutdown(self) -> None:
        """Disconnect and shutdown"""
        if self.connected:
            await self.disconnect()
            self.connected = False

        await super().shutdown()

    async def health_check(self) -> Dict[str, Any]:
        """Check integration health"""
        health = await super().health_check()

        # Test connection if connected
        if self.connected:
            try:
                if await self.test_connection():
                    self._clear_health_issues()
                else:
                    self._set_health_issue("Connection test failed", "warning")
            except Exception as e:
                self._set_health_issue(f"Health check error: {str(e)}", "error")

        health["connected"] = self.connected
        return health

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        status = super().get_status()
        status["connected"] = self.connected
        return status

    async def _handle_rate_limit(self) -> None:
        """Handle rate limiting"""
        if self.rate_limiter:
            await self.rate_limiter.acquire()

    async def _retry_with_backoff(self, func, max_retries: int = 3):
        """
        Retry function with exponential backoff.

        Args:
            func: Async function to retry
            max_retries: Maximum number of retries

        Returns:
            Function result
        """
        import asyncio

        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = 2**attempt
                self.logger.warning(
                    f"{self.name} attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
