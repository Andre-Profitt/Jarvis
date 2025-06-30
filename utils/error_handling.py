#!/usr/bin/env python3
"""
Comprehensive Error Handling for JARVIS
"""

import logging
import traceback
from functools import wraps
from typing import Any, Callable
import asyncio

logger = logging.getLogger(__name__)


def safe_execute(default_return=None):
    """Decorator for safe execution with error handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                return default_return

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class JARVISError(Exception):
    """Base exception for JARVIS"""

    pass


class ConfigurationError(JARVISError):
    """Configuration related errors"""

    pass


class IntegrationError(JARVISError):
    """AI integration errors"""

    pass


class DeviceError(JARVISError):
    """Device communication errors"""

    pass
