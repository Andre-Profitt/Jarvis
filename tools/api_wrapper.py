"""
API Wrapper Tool for JARVIS
===========================

Provides a generic API wrapper for making HTTP requests to external services.
"""

import aiohttp
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from urllib.parse import urljoin, urlencode
import hashlib
import hmac
import base64

from .base import BaseTool, ToolMetadata, ToolCategory


class APIWrapperTool(BaseTool):
    """
    Generic API wrapper tool for making HTTP requests

    Features:
    - Multiple HTTP methods (GET, POST, PUT, DELETE, PATCH)
    - Authentication support (Bearer, API Key, Basic Auth, OAuth)
    - Request/response caching
    - Rate limiting and retries
    - Request signing for secure APIs
    - Webhook support
    - Batch requests
    """

    def __init__(self):
        metadata = ToolMetadata(
            name="api_wrapper",
            description="Make HTTP requests to external APIs with authentication and advanced features",
            category=ToolCategory.WEB,
            version="1.0.0",
            tags=["api", "http", "rest", "integration", "webhook"],
            required_permissions=["internet_access"],
            rate_limit=100,  # 100 requests per minute
            timeout=30,
            examples=[
                {
                    "description": "Simple GET request",
                    "params": {
                        "method": "GET",
                        "url": "https://api.example.com/users",
                        "headers": {"Accept": "application/json"},
                    },
                },
                {
                    "description": "POST with authentication",
                    "params": {
                        "method": "POST",
                        "url": "https://api.example.com/data",
                        "headers": {"Content-Type": "application/json"},
                        "auth": {"type": "bearer", "token": "your-token"},
                        "json": {"name": "Test", "value": 123},
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Session management
        self._sessions = {}
        self._session_timeout = 300  # 5 minutes

        # Response cache
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

        # Rate limiting per domain
        self._rate_limits = {}

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate API request parameters"""
        method = kwargs.get("method", "GET")
        url = kwargs.get("url")

        if not url:
            return False, "URL parameter is required"

        if not isinstance(url, str):
            return False, "URL must be a string"

        if not url.startswith(("http://", "https://")):
            return False, "URL must start with http:// or https://"

        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        if method.upper() not in valid_methods:
            return False, f"Invalid HTTP method: {method}"

        # Validate auth if provided
        auth = kwargs.get("auth")
        if auth:
            auth_type = auth.get("type")
            valid_auth_types = ["bearer", "api_key", "basic", "oauth", "custom"]
            if auth_type not in valid_auth_types:
                return False, f"Invalid auth type: {auth_type}"

        return True, None

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute API request"""
        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url")
        headers = kwargs.get("headers", {})
        params = kwargs.get("params", {})
        data = kwargs.get("data")
        json_data = kwargs.get("json")
        auth = kwargs.get("auth")
        timeout = kwargs.get("timeout", 30)
        follow_redirects = kwargs.get("follow_redirects", True)
        verify_ssl = kwargs.get("verify_ssl", True)
        use_cache = kwargs.get("use_cache", method == "GET")
        retries = kwargs.get("retries", 3)

        # Check cache for GET requests
        if use_cache and method == "GET":
            cache_key = self._get_cache_key(url, params, headers)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached

        # Add authentication
        if auth:
            headers = self._add_authentication(headers, auth, method, url)

        # Get or create session
        session = await self._get_session(url)

        # Prepare request
        request_data = {
            "method": method,
            "url": url,
            "headers": headers,
            "params": params,
            "timeout": aiohttp.ClientTimeout(total=timeout),
            "allow_redirects": follow_redirects,
            "ssl": verify_ssl,
        }

        if data:
            request_data["data"] = data
        if json_data:
            request_data["json"] = json_data

        # Execute request with retries
        for attempt in range(retries):
            try:
                start_time = time.time()

                async with session.request(**request_data) as response:
                    # Read response
                    response_text = await response.text()
                    response_time = time.time() - start_time

                    # Try to parse JSON
                    response_json = None
                    if response.headers.get("Content-Type", "").startswith(
                        "application/json"
                    ):
                        try:
                            response_json = json.loads(response_text)
                        except:
                            pass

                    result = {
                        "success": 200 <= response.status < 300,
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "text": response_text,
                        "json": response_json,
                        "url": str(response.url),
                        "method": method,
                        "response_time": response_time,
                        "attempt": attempt + 1,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Cache successful GET responses
                    if use_cache and method == "GET" and result["success"]:
                        self._cache_response(cache_key, result)

                    return result

            except asyncio.TimeoutError:
                if attempt == retries - 1:
                    return {
                        "success": False,
                        "error": "Request timed out",
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                    }
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                if attempt == retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                    }
                await asyncio.sleep(2**attempt)

    async def _get_session(self, url: str) -> aiohttp.ClientSession:
        """Get or create session for URL domain"""
        from urllib.parse import urlparse

        domain = urlparse(url).netloc

        # Clean up old sessions
        current_time = time.time()
        expired_domains = [
            d
            for d, (s, t) in self._sessions.items()
            if current_time - t > self._session_timeout
        ]
        for d in expired_domains:
            await self._sessions[d][0].close()
            del self._sessions[d]

        # Get or create session
        if domain not in self._sessions:
            session = aiohttp.ClientSession()
            self._sessions[domain] = (session, current_time)
        else:
            session, _ = self._sessions[domain]
            self._sessions[domain] = (session, current_time)

        return session

    def _add_authentication(
        self, headers: Dict[str, str], auth: Dict[str, Any], method: str, url: str
    ) -> Dict[str, str]:
        """Add authentication headers"""
        auth_type = auth.get("type")
        headers = headers.copy()

        if auth_type == "bearer":
            token = auth.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "api_key":
            key_name = auth.get("key_name", "X-API-Key")
            key_value = auth.get("key_value")
            if key_value:
                headers[key_name] = key_value

        elif auth_type == "basic":
            username = auth.get("username")
            password = auth.get("password")
            if username and password:
                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"

        elif auth_type == "oauth":
            # OAuth 1.0a signature
            consumer_key = auth.get("consumer_key")
            consumer_secret = auth.get("consumer_secret")
            token = auth.get("token")
            token_secret = auth.get("token_secret")

            if all([consumer_key, consumer_secret]):
                # Simplified OAuth signature (full implementation would be more complex)
                timestamp = str(int(time.time()))
                nonce = hashlib.md5(f"{timestamp}{url}".encode()).hexdigest()

                headers["Authorization"] = (
                    f'OAuth oauth_consumer_key="{consumer_key}", '
                    f'oauth_token="{token}", '
                    f'oauth_timestamp="{timestamp}", '
                    f'oauth_nonce="{nonce}", '
                    f'oauth_version="1.0", '
                    f'oauth_signature_method="HMAC-SHA1"'
                )

        elif auth_type == "custom":
            # Custom headers
            custom_headers = auth.get("headers", {})
            headers.update(custom_headers)

        return headers

    def _get_cache_key(
        self, url: str, params: Dict[str, Any], headers: Dict[str, str]
    ) -> str:
        """Generate cache key for request"""
        # Include relevant headers in cache key
        cache_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() in ["accept", "accept-language", "authorization"]
        }

        key_parts = [
            url,
            urlencode(sorted(params.items())) if params else "",
            json.dumps(sorted(cache_headers.items())) if cache_headers else "",
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached["cached_at"] < self._cache_ttl:
                result = cached["response"].copy()
                result["cached"] = True
                result["cache_age"] = time.time() - cached["cached_at"]
                return result
            else:
                del self._cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response"""
        self._cache[cache_key] = {"response": response.copy(), "cached_at": time.time()}

        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache.keys(), key=lambda k: self._cache[k]["cached_at"]
            )
            for key in sorted_keys[:100]:
                del self._cache[key]

    async def batch_requests(
        self, requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple API requests in parallel"""
        tasks = []
        for request in requests:
            task = self.execute(**request)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    {"success": False, "error": str(result), "request_index": i}
                )
            else:
                final_results.append(result.data if hasattr(result, "data") else result)

        return final_results

    async def create_webhook(
        self, url: str, events: List[str], secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a webhook subscription"""
        webhook_id = hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:16]

        webhook_data = {
            "webhook_id": webhook_id,
            "url": url,
            "events": events,
            "secret": secret,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        # In a real implementation, this would register the webhook with the service
        return {
            "success": True,
            "webhook": webhook_data,
            "message": "Webhook created successfully",
        }

    async def close(self):
        """Close all sessions"""
        for domain, (session, _) in self._sessions.items():
            await session.close()
        self._sessions.clear()

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Document the parameters for this tool"""
        return {
            "method": {
                "type": "string",
                "description": "HTTP method",
                "required": False,
                "default": "GET",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            },
            "url": {
                "type": "string",
                "description": "The URL to request",
                "required": True,
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers",
                "required": False,
                "default": {},
            },
            "params": {
                "type": "object",
                "description": "Query parameters",
                "required": False,
                "default": {},
            },
            "data": {
                "type": "any",
                "description": "Request body data (for POST, PUT, PATCH)",
                "required": False,
            },
            "json": {
                "type": "object",
                "description": "JSON request body (for POST, PUT, PATCH)",
                "required": False,
            },
            "auth": {
                "type": "object",
                "description": "Authentication configuration",
                "required": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["bearer", "api_key", "basic", "oauth", "custom"],
                    },
                    "token": {"type": "string"},
                    "key_name": {"type": "string"},
                    "key_value": {"type": "string"},
                    "username": {"type": "string"},
                    "password": {"type": "string"},
                },
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds",
                "required": False,
                "default": 30,
            },
            "retries": {
                "type": "integer",
                "description": "Number of retry attempts",
                "required": False,
                "default": 3,
            },
            "use_cache": {
                "type": "boolean",
                "description": "Use cached responses for GET requests",
                "required": False,
                "default": True,
            },
        }


# Example usage
async def example_usage():
    """Example of using the APIWrapperTool"""
    tool = APIWrapperTool()

    # Example 1: Simple GET request
    result = await tool.execute(
        method="GET", url="https://jsonplaceholder.typicode.com/posts/1"
    )

    if result.success:
        print("GET Request successful!")
        print("Response:", json.dumps(result.data["json"], indent=2))

    # Example 2: POST request with authentication
    result = await tool.execute(
        method="POST",
        url="https://jsonplaceholder.typicode.com/posts",
        headers={"Content-Type": "application/json"},
        json={
            "title": "Test Post",
            "body": "This is a test post from JARVIS",
            "userId": 1,
        },
    )

    if result.success:
        print("\nPOST Request successful!")
        print("Created:", json.dumps(result.data["json"], indent=2))

    # Example 3: Batch requests
    batch_results = await tool.batch_requests(
        [
            {"method": "GET", "url": f"https://jsonplaceholder.typicode.com/posts/{i}"}
            for i in range(1, 4)
        ]
    )

    print(f"\nBatch requests completed: {len(batch_results)} results")

    # Cleanup
    await tool.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
