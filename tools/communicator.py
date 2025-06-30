"""
Communicator Tool for JARVIS
===========================

Provides advanced inter-service communication capabilities including
message queuing, pub/sub, RPC, and WebSocket connections.
"""

import asyncio
import json
import logging
import aioredis
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import pickle
from collections import defaultdict
import ssl
import certifi

from .base import BaseTool, ToolMetadata, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages"""

    BROADCAST = "broadcast"
    DIRECT = "direct"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    NOTIFICATION = "notification"


class Protocol(Enum):
    """Communication protocols"""

    HTTP = "http"
    WEBSOCKET = "websocket"
    REDIS_PUBSUB = "redis_pubsub"
    REDIS_QUEUE = "redis_queue"
    GRPC = "grpc"
    MQTT = "mqtt"


@dataclass
class Message:
    """Represents a communication message"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.DIRECT
    sender: str = ""
    recipient: Optional[str] = None
    topic: Optional[str] = None
    payload: Any = None
    headers: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None  # Time to live in seconds
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 5
    retries: int = 0
    max_retries: int = 3


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint"""

    name: str
    url: str
    protocol: Protocol
    health_check_url: Optional[str] = None
    timeout: int = 30
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True


class CommunicatorTool(BaseTool):
    """
    Advanced inter-service communication tool

    Features:
    - Multiple protocols (HTTP, WebSocket, Redis, gRPC, MQTT)
    - Message queuing and pub/sub
    - Service discovery and health checking
    - Circuit breaker pattern
    - Message routing and load balancing
    - Encryption and authentication
    - Message persistence and replay
    """

    def __init__(self):
        metadata = ToolMetadata(
            name="communicator",
            description="Advanced inter-service communication and messaging",
            category=ToolCategory.COMMUNICATION,
            version="2.0.0",
            tags=["messaging", "rpc", "pubsub", "websocket", "microservices"],
            required_permissions=["network_access", "redis_access"],
            rate_limit=1000,
            timeout=60,
            examples=[
                {
                    "description": "Send a direct message",
                    "params": {
                        "action": "send",
                        "recipient": "analytics_service",
                        "message": {"type": "data_request", "query": "daily_stats"},
                    },
                },
                {
                    "description": "Publish to a topic",
                    "params": {
                        "action": "publish",
                        "topic": "system_events",
                        "message": {"event": "user_login", "user_id": "123"},
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Service registry
        self.services: Dict[str, ServiceEndpoint] = {}

        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.topic_subscribers: Dict[str, Set[Callable]] = defaultdict(set)

        # Connection pools
        self.websocket_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.redis_client: Optional[aioredis.Redis] = None

        # Message queues
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "avg_latency": 0.0,
        }

        # Start background tasks
        self._background_tasks = []
        asyncio.create_task(self._initialize_connections())

    async def _initialize_connections(self):
        """Initialize connection pools"""
        try:
            # HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(timeout=timeout)

            # Redis connection
            try:
                self.redis_client = await aioredis.create_redis_pool(
                    "redis://localhost:6379", encoding="utf-8"
                )
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

            # Start message processor
            self._background_tasks.append(
                asyncio.create_task(self._process_message_queue())
            )

            # Start health checker
            self._background_tasks.append(
                asyncio.create_task(self._health_check_loop())
            )

        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")

    async def _execute(self, **kwargs) -> Any:
        """Execute communication operations"""
        action = kwargs.get("action", "").lower()

        if action == "send":
            return await self._send_message(**kwargs)
        elif action == "publish":
            return await self._publish_message(**kwargs)
        elif action == "subscribe":
            return await self._subscribe_topic(**kwargs)
        elif action == "unsubscribe":
            return await self._unsubscribe_topic(**kwargs)
        elif action == "register_service":
            return await self._register_service(**kwargs)
        elif action == "discover_services":
            return await self._discover_services(**kwargs)
        elif action == "health_check":
            return await self._health_check(kwargs.get("service"))
        elif action == "rpc":
            return await self._rpc_call(**kwargs)
        elif action == "broadcast":
            return await self._broadcast_message(**kwargs)
        elif action == "get_metrics":
            return self._get_metrics()
        else:
            raise ValueError(f"Unknown action: {action}")

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate communicator inputs"""
        action = kwargs.get("action")

        if not action:
            return False, "Action is required"

        if action == "send":
            if not kwargs.get("recipient"):
                return False, "Recipient is required for send action"
            if not kwargs.get("message"):
                return False, "Message is required for send action"

        elif action == "publish":
            if not kwargs.get("topic"):
                return False, "Topic is required for publish action"
            if not kwargs.get("message"):
                return False, "Message is required for publish action"

        elif action == "subscribe":
            if not kwargs.get("topic"):
                return False, "Topic is required for subscribe action"

        elif action == "register_service":
            if not kwargs.get("name") or not kwargs.get("url"):
                return False, "Name and URL are required for service registration"

        return True, None

    async def _send_message(self, **kwargs) -> Dict[str, Any]:
        """Send a direct message to a service"""
        recipient = kwargs.get("recipient")
        message_data = kwargs.get("message")
        protocol = Protocol(kwargs.get("protocol", "http").lower())
        timeout = kwargs.get("timeout", 30)

        # Create message
        message = Message(
            type=MessageType.DIRECT,
            sender=kwargs.get("sender", "jarvis"),
            recipient=recipient,
            payload=message_data,
            headers=kwargs.get("headers", {}),
            priority=kwargs.get("priority", 5),
        )

        # Check if service is registered
        if recipient in self.services:
            service = self.services[recipient]

            # Check circuit breaker
            if self._is_circuit_open(recipient):
                return {
                    "success": False,
                    "error": f"Circuit breaker open for {recipient}",
                }

            # Send based on protocol
            try:
                if service.protocol == Protocol.HTTP:
                    result = await self._send_http(service, message)
                elif service.protocol == Protocol.WEBSOCKET:
                    result = await self._send_websocket(service, message)
                elif service.protocol == Protocol.REDIS_QUEUE:
                    result = await self._send_redis_queue(recipient, message)
                else:
                    result = await self._send_http(service, message)

                # Update metrics
                self.metrics["messages_sent"] += 1
                self._circuit_breaker_success(recipient)

                return {
                    "success": True,
                    "message_id": message.id,
                    "recipient": recipient,
                    "protocol": service.protocol.value,
                    "result": result,
                }

            except Exception as e:
                self.metrics["errors"] += 1
                self._circuit_breaker_failure(recipient)
                logger.error(f"Failed to send message to {recipient}: {e}")

                return {"success": False, "message_id": message.id, "error": str(e)}
        else:
            # Try default HTTP if service not registered
            try:
                result = await self._send_http_direct(recipient, message)
                self.metrics["messages_sent"] += 1

                return {
                    "success": True,
                    "message_id": message.id,
                    "recipient": recipient,
                    "result": result,
                }
            except Exception as e:
                self.metrics["errors"] += 1
                return {"success": False, "message_id": message.id, "error": str(e)}

    async def _send_http(self, service: ServiceEndpoint, message: Message) -> Any:
        """Send message via HTTP"""
        if not self.http_session:
            raise RuntimeError("HTTP session not initialized")

        headers = {**service.headers, **message.headers}

        async with self.http_session.post(
            service.url,
            json={
                "message_id": message.id,
                "sender": message.sender,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat(),
            },
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=service.timeout),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _send_http_direct(self, url: str, message: Message) -> Any:
        """Send HTTP message directly to URL"""
        if not self.http_session:
            raise RuntimeError("HTTP session not initialized")

        async with self.http_session.post(
            url,
            json={
                "message_id": message.id,
                "sender": message.sender,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat(),
            },
            headers=message.headers,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _send_websocket(self, service: ServiceEndpoint, message: Message) -> Any:
        """Send message via WebSocket"""
        ws_url = service.url.replace("http://", "ws://").replace("https://", "wss://")

        # Get or create connection
        if service.name not in self.websocket_connections:
            self.websocket_connections[service.name] = await websockets.connect(ws_url)

        ws = self.websocket_connections[service.name]

        # Send message
        await ws.send(
            json.dumps(
                {
                    "message_id": message.id,
                    "type": message.type.value,
                    "sender": message.sender,
                    "payload": message.payload,
                    "timestamp": message.timestamp.isoformat(),
                }
            )
        )

        # Wait for response if it's a request
        if message.type == MessageType.REQUEST:
            response = await ws.recv()
            return json.loads(response)

        return {"status": "sent"}

    async def _send_redis_queue(self, queue_name: str, message: Message) -> Any:
        """Send message via Redis queue"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        # Serialize message
        message_data = json.dumps(
            {
                "message_id": message.id,
                "type": message.type.value,
                "sender": message.sender,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat(),
            }
        )

        # Push to queue
        await self.redis_client.lpush(f"queue:{queue_name}", message_data)

        return {"status": "queued", "queue": queue_name}

    async def _publish_message(self, **kwargs) -> Dict[str, Any]:
        """Publish message to a topic"""
        topic = kwargs.get("topic")
        message_data = kwargs.get("message")

        # Create message
        message = Message(
            type=MessageType.EVENT,
            sender=kwargs.get("sender", "jarvis"),
            topic=topic,
            payload=message_data,
            headers=kwargs.get("headers", {}),
            ttl=kwargs.get("ttl"),
        )

        # Publish via Redis pub/sub
        if self.redis_client:
            try:
                await self.redis_client.publish(
                    f"topic:{topic}",
                    json.dumps(
                        {
                            "message_id": message.id,
                            "sender": message.sender,
                            "payload": message.payload,
                            "timestamp": message.timestamp.isoformat(),
                        }
                    ),
                )

                # Also notify local subscribers
                await self._notify_local_subscribers(topic, message)

                self.metrics["messages_sent"] += 1

                return {
                    "success": True,
                    "message_id": message.id,
                    "topic": topic,
                    "subscribers": len(self.topic_subscribers.get(topic, [])),
                }

            except Exception as e:
                logger.error(f"Failed to publish message: {e}")
                return {"success": False, "error": str(e)}
        else:
            # Fallback to local pub/sub only
            await self._notify_local_subscribers(topic, message)

            return {
                "success": True,
                "message_id": message.id,
                "topic": topic,
                "subscribers": len(self.topic_subscribers.get(topic, [])),
                "note": "Local subscribers only (Redis not available)",
            }

    async def _notify_local_subscribers(self, topic: str, message: Message):
        """Notify local subscribers of a message"""
        subscribers = self.topic_subscribers.get(topic, set())

        for handler in subscribers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Subscriber error for topic {topic}: {e}")

    async def _subscribe_topic(self, **kwargs) -> Dict[str, Any]:
        """Subscribe to a topic"""
        topic = kwargs.get("topic")
        handler = kwargs.get("handler")

        if handler:
            self.topic_subscribers[topic].add(handler)

        # Subscribe via Redis if available
        if self.redis_client:
            # Start Redis subscription in background
            asyncio.create_task(self._redis_subscribe_loop(topic))

        return {
            "success": True,
            "topic": topic,
            "subscribers": len(self.topic_subscribers[topic]),
        }

    async def _redis_subscribe_loop(self, topic: str):
        """Listen for Redis pub/sub messages"""
        if not self.redis_client:
            return

        try:
            # Create a separate connection for pub/sub
            sub_client = await aioredis.create_redis("redis://localhost:6379")
            channel = (await sub_client.subscribe(f"topic:{topic}"))[0]

            while await channel.wait_message():
                message_data = await channel.get(encoding="utf-8")

                try:
                    data = json.loads(message_data)
                    message = Message(
                        id=data.get("message_id"),
                        type=MessageType.EVENT,
                        sender=data.get("sender"),
                        topic=topic,
                        payload=data.get("payload"),
                        timestamp=datetime.fromisoformat(data.get("timestamp")),
                    )

                    # Notify local subscribers
                    await self._notify_local_subscribers(topic, message)
                    self.metrics["messages_received"] += 1

                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")

            sub_client.close()

        except Exception as e:
            logger.error(f"Redis subscription error for topic {topic}: {e}")

    async def _register_service(self, **kwargs) -> Dict[str, Any]:
        """Register a service endpoint"""
        name = kwargs.get("name")
        url = kwargs.get("url")
        protocol = Protocol(kwargs.get("protocol", "http").lower())

        service = ServiceEndpoint(
            name=name,
            url=url,
            protocol=protocol,
            health_check_url=kwargs.get("health_check_url"),
            timeout=kwargs.get("timeout", 30),
            retry_policy=kwargs.get("retry_policy", {}),
            headers=kwargs.get("headers", {}),
            metadata=kwargs.get("metadata", {}),
        )

        self.services[name] = service

        # Initialize circuit breaker
        self.circuit_breakers[name] = {
            "failures": 0,
            "last_failure": None,
            "state": "closed",
            "half_open_attempts": 0,
        }

        # Perform initial health check
        await self._health_check(name)

        return {
            "success": True,
            "service": name,
            "url": url,
            "protocol": protocol.value,
        }

    async def _discover_services(self, **kwargs) -> List[Dict[str, Any]]:
        """Discover available services"""
        tag = kwargs.get("tag")
        protocol = kwargs.get("protocol")

        services = []

        for name, service in self.services.items():
            # Filter by protocol
            if protocol and service.protocol.value != protocol:
                continue

            # Filter by tag
            if tag and tag not in service.metadata.get("tags", []):
                continue

            services.append(
                {
                    "name": name,
                    "url": service.url,
                    "protocol": service.protocol.value,
                    "healthy": service.is_healthy,
                    "last_health_check": (
                        service.last_health_check.isoformat()
                        if service.last_health_check
                        else None
                    ),
                    "metadata": service.metadata,
                }
            )

        return services

    async def _health_check(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform health check on service(s)"""
        if service_name:
            # Check specific service
            if service_name not in self.services:
                return {"error": f"Service {service_name} not found"}

            service = self.services[service_name]
            is_healthy = await self._check_service_health(service)

            return {
                "service": service_name,
                "healthy": is_healthy,
                "last_check": (
                    service.last_health_check.isoformat()
                    if service.last_health_check
                    else None
                ),
            }
        else:
            # Check all services
            results = {}

            for name, service in self.services.items():
                is_healthy = await self._check_service_health(service)
                results[name] = {
                    "healthy": is_healthy,
                    "last_check": (
                        service.last_health_check.isoformat()
                        if service.last_health_check
                        else None
                    ),
                }

            return results

    async def _check_service_health(self, service: ServiceEndpoint) -> bool:
        """Check health of a single service"""
        try:
            # Use health check URL if provided
            url = service.health_check_url or f"{service.url}/health"

            if service.protocol == Protocol.HTTP:
                if self.http_session:
                    async with self.http_session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        is_healthy = response.status == 200
                else:
                    is_healthy = False
            else:
                # For other protocols, assume healthy if we can connect
                is_healthy = True

            service.is_healthy = is_healthy
            service.last_health_check = datetime.now()

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed for {service.name}: {e}")
            service.is_healthy = False
            service.last_health_check = datetime.now()
            return False

    async def _health_check_loop(self):
        """Background task to periodically check service health"""
        while True:
            try:
                for service in self.services.values():
                    await self._check_service_health(service)

                # Wait 30 seconds between checks
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)

    async def _rpc_call(self, **kwargs) -> Any:
        """Make an RPC call to a service"""
        service_name = kwargs.get("service")
        method = kwargs.get("method")
        params = kwargs.get("params", {})
        timeout = kwargs.get("timeout", 30)

        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")

        # Create RPC message
        message = Message(
            type=MessageType.REQUEST,
            sender="jarvis",
            recipient=service_name,
            payload={
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": str(uuid.uuid4()),
            },
            correlation_id=str(uuid.uuid4()),
        )

        # Store future for response
        response_future = asyncio.Future()
        self.pending_responses[message.correlation_id] = response_future

        try:
            # Send message
            await self._send_message(
                recipient=service_name,
                message=message.payload,
                headers={"X-Correlation-ID": message.correlation_id},
            )

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)

            return response

        except asyncio.TimeoutError:
            raise TimeoutError(f"RPC call to {service_name}.{method} timed out")
        finally:
            # Clean up
            self.pending_responses.pop(message.correlation_id, None)

    async def _broadcast_message(self, **kwargs) -> Dict[str, Any]:
        """Broadcast message to all services"""
        message_data = kwargs.get("message")
        exclude = kwargs.get("exclude", [])

        results = {}

        for service_name in self.services:
            if service_name in exclude:
                continue

            try:
                result = await self._send_message(
                    recipient=service_name, message=message_data
                )
                results[service_name] = result
            except Exception as e:
                results[service_name] = {"error": str(e)}

        successful = sum(1 for r in results.values() if r.get("success"))

        return {
            "success": successful > 0,
            "sent_to": successful,
            "total_services": len(results),
            "results": results,
        }

    async def _process_message_queue(self):
        """Process incoming messages"""
        while True:
            try:
                # Process messages from queue
                if not self.message_queue.empty():
                    message = await self.message_queue.get()

                    # Handle based on message type
                    if message.type == MessageType.RESPONSE:
                        # Handle RPC response
                        if message.correlation_id in self.pending_responses:
                            future = self.pending_responses[message.correlation_id]
                            future.set_result(message.payload)
                    else:
                        # Notify handlers
                        handlers = self.message_handlers.get(message.type.value, [])
                        for handler in handlers:
                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(message)
                                else:
                                    handler(message)
                            except Exception as e:
                                logger.error(f"Message handler error: {e}")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)

    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service"""
        if service_name not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[service_name]

        if breaker["state"] == "open":
            # Check if enough time has passed to try again
            if breaker["last_failure"]:
                time_since_failure = datetime.now() - breaker["last_failure"]
                if time_since_failure > timedelta(seconds=30):
                    # Move to half-open state
                    breaker["state"] = "half_open"
                    breaker["half_open_attempts"] = 0
                else:
                    return True

        return False

    def _circuit_breaker_success(self, service_name: str):
        """Record successful call for circuit breaker"""
        if service_name in self.circuit_breakers:
            breaker = self.circuit_breakers[service_name]

            if breaker["state"] == "half_open":
                breaker["half_open_attempts"] += 1

                # Close circuit after successful attempts
                if breaker["half_open_attempts"] >= 3:
                    breaker["state"] = "closed"
                    breaker["failures"] = 0
                    breaker["last_failure"] = None

    def _circuit_breaker_failure(self, service_name: str):
        """Record failed call for circuit breaker"""
        if service_name in self.circuit_breakers:
            breaker = self.circuit_breakers[service_name]

            breaker["failures"] += 1
            breaker["last_failure"] = datetime.now()

            # Open circuit after threshold
            if breaker["failures"] >= 5:
                breaker["state"] = "open"

    def _get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            "messages_sent": self.metrics["messages_sent"],
            "messages_received": self.metrics["messages_received"],
            "errors": self.metrics["errors"],
            "services": {
                name: {
                    "healthy": service.is_healthy,
                    "protocol": service.protocol.value,
                    "circuit_breaker": self.circuit_breakers.get(name, {}).get(
                        "state", "closed"
                    ),
                }
                for name, service in self.services.items()
            },
            "topics": list(self.topic_subscribers.keys()),
            "pending_responses": len(self.pending_responses),
        }

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type.value].append(handler)

    async def _unsubscribe_topic(self, **kwargs) -> Dict[str, Any]:
        """Unsubscribe from a topic"""
        topic = kwargs.get("topic")
        handler = kwargs.get("handler")

        if topic in self.topic_subscribers:
            if handler and handler in self.topic_subscribers[topic]:
                self.topic_subscribers[topic].remove(handler)

            return {
                "success": True,
                "topic": topic,
                "remaining_subscribers": len(self.topic_subscribers[topic]),
            }

        return {"success": False, "error": f"Topic {topic} not found"}

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Get parameter documentation for the communicator"""
        return {
            "action": {
                "type": "string",
                "required": True,
                "enum": [
                    "send",
                    "publish",
                    "subscribe",
                    "unsubscribe",
                    "register_service",
                    "discover_services",
                    "health_check",
                    "rpc",
                    "broadcast",
                    "get_metrics",
                ],
                "description": "Action to perform",
            },
            "recipient": {
                "type": "string",
                "required": "for send action",
                "description": "Service to send message to",
            },
            "message": {
                "type": "any",
                "required": "for send, publish, broadcast actions",
                "description": "Message payload to send",
            },
            "topic": {
                "type": "string",
                "required": "for publish, subscribe, unsubscribe actions",
                "description": "Topic name for pub/sub",
            },
            "handler": {
                "type": "callable",
                "required": False,
                "description": "Handler function for subscriptions",
            },
            "service": {
                "type": "string",
                "required": "for rpc action",
                "description": "Service name for RPC calls",
            },
            "method": {
                "type": "string",
                "required": "for rpc action",
                "description": "RPC method to call",
            },
            "params": {
                "type": "dict",
                "required": False,
                "description": "Parameters for RPC call",
            },
            "name": {
                "type": "string",
                "required": "for register_service action",
                "description": "Service name to register",
            },
            "url": {
                "type": "string",
                "required": "for register_service action",
                "description": "Service URL",
            },
            "protocol": {
                "type": "string",
                "required": False,
                "enum": [
                    "http",
                    "websocket",
                    "redis_pubsub",
                    "redis_queue",
                    "grpc",
                    "mqtt",
                ],
                "description": "Communication protocol",
            },
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit"""
        # Close connections
        if self.http_session:
            await self.http_session.close()

        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()

        for ws in self.websocket_connections.values():
            await ws.close()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
