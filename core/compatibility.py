#!/usr/bin/env python3
"""
Compatibility wrapper for database methods
Ensures both old and new APIs work
"""

from functools import wraps
import warnings

def compatibility_wrapper(old_params=None):
    """Decorator to handle parameter compatibility"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract old parameter names
            if old_params:
                for old, new in old_params.items():
                    if old in kwargs and new not in kwargs:
                        kwargs[new] = kwargs.pop(old)
                        warnings.warn(
                            f"Parameter '{old}' is deprecated, use '{new}'",
                            DeprecationWarning
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage:
# @compatibility_wrapper({'user_id': 'context.user_id'})
# def store_memory(self, content, memory_type='general', context=None):
#     # Implementation
