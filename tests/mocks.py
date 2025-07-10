"""
JARVIS Test Mocks
=================
Comprehensive mock objects for all external dependencies and services.
"""
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime


# ===== AI Service Mocks =====
class MockOpenAI:
    """Mock OpenAI API client"""
    def __init__(self):
        self.chat = Mock()
        self.embeddings = Mock()
        self.images = Mock()
        self.audio = Mock()
        self.completions = Mock()
        
        # Setup chat completions
        self.chat.completions.create = AsyncMock(side_effect=self._chat_completion)
        
        # Setup embeddings
        self.embeddings.create = AsyncMock(side_effect=self._create_embedding)
        
        # Track API calls
        self.call_count = 0
        self.last_prompt = None
    
    async def _chat_completion(self, **kwargs):
        """Simulate chat completion"""
        self.call_count += 1
        self.last_prompt = kwargs.get('messages', [])
        
        return Mock(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            choices=[
                Mock(
                    message=Mock(
                        content="This is a mocked OpenAI response for testing.",
                        role="assistant"
                    ),
                    finish_reason="stop",
                    index=0
                )
            ],
            created=int(datetime.now().timestamp()),
            model=kwargs.get('model', 'gpt-4'),
            usage=Mock(
                prompt_tokens=50,
                completion_tokens=25,
                total_tokens=75
            )
        )
    
    async def _create_embedding(self, **kwargs):
        """Simulate embedding creation"""
        input_text = kwargs.get('input', '')
        # Return consistent embedding for testing
        embedding = [0.1] * 1536  # OpenAI embedding dimension
        
        return Mock(
            data=[Mock(embedding=embedding, index=0)],
            model='text-embedding-ada-002',
            usage=Mock(prompt_tokens=len(input_text.split()))
        )


class MockAnthropic:
    """Mock Anthropic Claude API client"""
    def __init__(self):
        self.messages = Mock()
        self.messages.create = AsyncMock(side_effect=self._create_message)
        self.call_count = 0
        self.last_messages = None
    
    async def _create_message(self, **kwargs):
        """Simulate Claude message creation"""
        self.call_count += 1
        self.last_messages = kwargs.get('messages', [])
        
        return Mock(
            id=f"msg-{uuid.uuid4().hex[:8]}",
            content=[
                Mock(
                    text="This is a mocked Claude response for testing.",
                    type="text"
                )
            ],
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            role="assistant",
            stop_reason="end_turn",
            usage=Mock(
                input_tokens=100,
                output_tokens=50
            )
        )


class MockGemini:
    """Mock Google Gemini API client"""
    def __init__(self):
        self.generate_content = AsyncMock(side_effect=self._generate_content)
        self.call_count = 0
    
    async def _generate_content(self, prompt: str, **kwargs):
        """Simulate Gemini content generation"""
        self.call_count += 1
        
        return Mock(
            text="This is a mocked Gemini response for testing.",
            candidates=[
                Mock(
                    content=Mock(
                        parts=[Mock(text="This is a mocked Gemini response for testing.")],
                        role="model"
                    ),
                    finish_reason="STOP"
                )
            ],
            prompt_feedback=Mock(safety_ratings=[])
        )


# ===== Database and Cache Mocks =====
class MockRedis:
    """In-memory Redis mock for testing"""
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        self.pubsub_channels: Dict[str, List[callable]] = {}
        
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        return self.data.get(key)
    
    async def set(self, key: str, value: Any, ex: int = None) -> None:
        """Set value with optional expiry"""
        self.data[key] = value if isinstance(value, str) else json.dumps(value)
        if ex:
            self.expiry[key] = datetime.now().timestamp() + ex
    
    async def delete(self, *keys: str) -> int:
        """Delete keys"""
        deleted = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                self.expiry.pop(key, None)
                deleted += 1
        return deleted
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        return sum(1 for key in keys if key in self.data)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiry on key"""
        if key in self.data:
            self.expiry[key] = datetime.now().timestamp() + seconds
            return True
        return False
    
    async def ttl(self, key: str) -> int:
        """Get time to live for key"""
        if key not in self.expiry:
            return -1 if key in self.data else -2
        
        ttl = int(self.expiry[key] - datetime.now().timestamp())
        return max(0, ttl)
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        if pattern == "*":
            return list(self.data.keys())
        
        # Simple pattern matching
        import re
        regex = pattern.replace("*", ".*").replace("?", ".")
        return [k for k in self.data.keys() if re.match(f"^{regex}$", k)]
    
    async def hset(self, name: str, key: str, value: Any) -> int:
        """Set hash field"""
        if name not in self.data:
            self.data[name] = {}
        self.data[name][key] = value
        return 1
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field"""
        if name in self.data and isinstance(self.data[name], dict):
            return self.data[name].get(key)
        return None
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields"""
        if name in self.data and isinstance(self.data[name], dict):
            return self.data[name]
        return {}
    
    async def flushall(self) -> None:
        """Clear all data"""
        self.data.clear()
        self.expiry.clear()
    
    def pubsub(self):
        """Get pubsub interface"""
        return MockRedisPubSub(self)


class MockRedisPubSub:
    """Mock Redis PubSub"""
    def __init__(self, redis: MockRedis):
        self.redis = redis
        self.subscriptions: Dict[str, bool] = {}
        
    async def subscribe(self, *channels: str) -> None:
        """Subscribe to channels"""
        for channel in channels:
            self.subscriptions[channel] = True
            if channel not in self.redis.pubsub_channels:
                self.redis.pubsub_channels[channel] = []
    
    async def unsubscribe(self, *channels: str) -> None:
        """Unsubscribe from channels"""
        for channel in channels:
            self.subscriptions.pop(channel, None)
    
    async def get_message(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Get pubsub message"""
        # Return None for testing
        return None


class MockDatabaseManager:
    """Mock database manager"""
    def __init__(self, connection_string: str = ":memory:"):
        self.connection_string = connection_string
        self.data: Dict[str, List[Dict]] = {}
        self.is_connected = False
        
    async def initialize(self) -> None:
        """Initialize database"""
        self.is_connected = True
        
    async def close(self) -> None:
        """Close database"""
        self.is_connected = False
        
    async def execute(self, query: str, params: tuple = None) -> Any:
        """Execute query"""
        # Simple mock implementation
        if "CREATE TABLE" in query:
            table_name = query.split("CREATE TABLE")[1].split("(")[0].strip()
            self.data[table_name] = []
            return None
        elif "INSERT INTO" in query:
            table_name = query.split("INSERT INTO")[1].split("(")[0].strip()
            if table_name not in self.data:
                self.data[table_name] = []
            self.data[table_name].append({"id": len(self.data[table_name]) + 1})
            return Mock(lastrowid=len(self.data[table_name]))
        elif "SELECT" in query:
            table_name = query.split("FROM")[1].split()[0].strip()
            return self.data.get(table_name, [])
        
        return None
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Fetch one row"""
        results = await self.execute(query, params)
        return results[0] if results else None
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """Fetch all rows"""
        return await self.execute(query, params) or []


# ===== Service Mocks =====
class MockElevenLabs:
    """Mock ElevenLabs TTS API"""
    def __init__(self):
        self.generate = AsyncMock(side_effect=self._generate_audio)
        self.voices = Mock()
        self.voices.get_all = AsyncMock(return_value=[
            {"voice_id": "test_voice_1", "name": "Test Voice 1"},
            {"voice_id": "test_voice_2", "name": "Test Voice 2"}
        ])
    
    async def _generate_audio(self, text: str, voice: str = None, **kwargs):
        """Generate mock audio"""
        # Return mock audio bytes
        return b"MOCK_AUDIO_DATA" * 100


class MockWebSocket:
    """Mock WebSocket connection"""
    def __init__(self):
        self.messages_to_receive = []
        self.sent_messages = []
        self.is_closed = False
        
    async def send(self, message: str) -> None:
        """Send message"""
        if self.is_closed:
            raise RuntimeError("WebSocket is closed")
        self.sent_messages.append(message)
        
    async def recv(self) -> str:
        """Receive message"""
        if self.is_closed:
            raise RuntimeError("WebSocket is closed")
        if self.messages_to_receive:
            return self.messages_to_receive.pop(0)
        # Default test message
        return json.dumps({"type": "ping", "data": {"timestamp": datetime.now().isoformat()}})
        
    async def close(self) -> None:
        """Close connection"""
        self.is_closed = True
        
    def add_message_to_receive(self, message: Dict[str, Any]) -> None:
        """Add message to receive queue for testing"""
        self.messages_to_receive.append(json.dumps(message))


# ===== Agent and Component Mocks =====
class MockAgent:
    """Base mock agent for testing"""
    def __init__(self, name: str, capabilities: List[str] = None):
        self.name = name
        self.capabilities = capabilities or ["general"]
        self.is_available = True
        self.execution_history = []
        
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task"""
        result = {
            "status": "success",
            "agent": self.name,
            "task_id": task.get("id", "unknown"),
            "result": f"Mock result from {self.name}",
            "timestamp": datetime.now().isoformat()
        }
        self.execution_history.append(result)
        return result
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities
    
    def set_availability(self, available: bool) -> None:
        """Set agent availability"""
        self.is_available = available


class MockAnalyzerAgent(MockAgent):
    """Mock analyzer agent"""
    def __init__(self):
        super().__init__("AnalyzerAgent", ["code_analysis", "pattern_detection"])
        
    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code"""
        return {
            "complexity": "medium",
            "issues": [],
            "suggestions": ["Consider adding type hints"],
            "metrics": {
                "lines": len(code.split("\n")),
                "functions": 2,
                "classes": 1
            }
        }


class MockQAAgent(MockAgent):
    """Mock QA agent"""
    def __init__(self):
        super().__init__("QAAgent", ["testing", "validation", "quality_assurance"])
        
    async def run_tests(self, test_suite: str) -> Dict[str, Any]:
        """Run tests"""
        return {
            "passed": 10,
            "failed": 0,
            "skipped": 2,
            "coverage": 85.5,
            "duration": 1.23
        }


class MockToolExecutor:
    """Mock tool executor"""
    def __init__(self):
        self.available_tools = {
            "calculator": self._calculator,
            "web_search": self._web_search,
            "file_reader": self._file_reader
        }
        self.execution_log = []
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute tool"""
        if tool_name not in self.available_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        result = await self.available_tools[tool_name](params)
        self.execution_log.append({
            "tool": tool_name,
            "params": params,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        return result
    
    async def _calculator(self, params: Dict[str, Any]) -> float:
        """Mock calculator tool"""
        expression = params.get("expression", "0")
        # Safe evaluation for testing - only basic math
        try:
            # Parse simple math expressions safely
            import ast
            import operator
            
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow
            }
            
            def eval_expr(expr):
                if isinstance(expr, ast.Num):
                    return expr.n
                elif isinstance(expr, ast.BinOp):
                    return ops[type(expr.op)](eval_expr(expr.left), eval_expr(expr.right))
                else:
                    raise ValueError("Unsupported expression")
            
            tree = ast.parse(expression, mode='eval')
            return eval_expr(tree.body)
        except:
            return 0.0
    
    async def _web_search(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Mock web search tool"""
        query = params.get("query", "")
        return [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1", "snippet": "Test result 1"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2", "snippet": "Test result 2"}
        ]
    
    async def _file_reader(self, params: Dict[str, Any]) -> str:
        """Mock file reader tool"""
        filename = params.get("filename", "test.txt")
        return f"Mock content of {filename}"


# ===== Utility Functions =====
def create_mock_jarvis_response(content: str = "Mock JARVIS response") -> Dict[str, Any]:
    """Create a mock JARVIS response"""
    return {
        "response": content,
        "metadata": {
            "model": "jarvis-advanced",
            "tokens_used": 150,
            "processing_time": 0.234,
            "confidence": 0.95
        },
        "timestamp": datetime.now().isoformat()
    }


def create_mock_task_result(task_id: str, status: str = "success") -> Dict[str, Any]:
    """Create a mock task result"""
    return {
        "task_id": task_id,
        "status": status,
        "result": f"Mock result for task {task_id}",
        "execution_time": 1.23,
        "resources_used": {
            "cpu": 45.2,
            "memory": 128.5,
            "gpu": 0.0
        },
        "timestamp": datetime.now().isoformat()
    }


def create_mock_websocket_message(msg_type: str, data: Dict[str, Any] = None) -> str:
    """Create a mock WebSocket message"""
    return json.dumps({
        "type": msg_type,
        "data": data or {},
        "timestamp": datetime.now().isoformat()
    })


# ===== Test Data Generators =====
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def create_conversation(num_messages: int = 5) -> List[Dict[str, str]]:
        """Create a test conversation"""
        messages = []
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Test message {i + 1} from {role}"
            })
        return messages
    
    @staticmethod
    def create_user_profile(user_id: str = None) -> Dict[str, Any]:
        """Create a test user profile"""
        return {
            "user_id": user_id or f"user_{uuid.uuid4().hex[:8]}",
            "name": "Test User",
            "preferences": {
                "language": "en",
                "theme": "dark",
                "notifications": True
            },
            "created_at": datetime.now().isoformat(),
            "subscription": "premium"
        }
    
    @staticmethod
    def create_system_metrics() -> Dict[str, Any]:
        """Create test system metrics"""
        return {
            "cpu_usage": 42.5,
            "memory_usage": 65.3,
            "active_connections": 127,
            "requests_per_second": 1543,
            "average_response_time": 0.145,
            "error_rate": 0.02,
            "timestamp": datetime.now().isoformat()
        }