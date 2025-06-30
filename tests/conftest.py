"""
JARVIS Test Configuration and Shared Fixtures
============================================
This module provides shared test fixtures and configuration for the entire test suite.
"""
import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import json
import tempfile
from typing import Dict, Any, Generator, AsyncGenerator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ['JARVIS_ENV'] = 'test'
os.environ['TESTING'] = 'true'
os.environ['OPENAI_API_KEY'] = 'test-openai-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
os.environ['ELEVENLABS_API_KEY'] = 'test-elevenlabs-key'
os.environ['GEMINI_API_KEY'] = 'test-gemini-key'


# ===== Event Loop Configuration =====
@pytest.fixture(scope="session")
def event_loop_policy():
    """Create event loop policy for async tests"""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def event_loop(event_loop_policy):
    """Create event loop for each test"""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


# ===== AI Client Mocks =====
@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch('openai.OpenAI') as mock_client:
        instance = Mock()
        mock_client.return_value = instance
        
        # Mock chat completions
        instance.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Test OpenAI response"))],
                usage=Mock(total_tokens=100)
            )
        )
        
        # Mock embeddings
        instance.embeddings.create = AsyncMock(
            return_value=Mock(
                data=[Mock(embedding=[0.1] * 1536)]
            )
        )
        
        yield instance


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client"""
    with patch('anthropic.Anthropic') as mock_client:
        instance = Mock()
        mock_client.return_value = instance
        
        # Mock messages
        instance.messages.create = AsyncMock(
            return_value=Mock(
                content=[Mock(text="Test Claude response")],
                usage=Mock(input_tokens=50, output_tokens=50)
            )
        )
        
        yield instance


@pytest.fixture
def mock_all_ai_clients(mock_openai, mock_anthropic):
    """Mock all AI clients at once"""
    return {
        'openai': mock_openai,
        'anthropic': mock_anthropic,
    }


# ===== Database Fixtures =====
@pytest.fixture
async def mock_database():
    """Provide in-memory database for testing"""
    from core.database import DatabaseManager
    
    # Use in-memory SQLite for tests
    db = DatabaseManager(":memory:")
    
    yield db
    
    # No need to close as SQLite in-memory DB is automatically cleaned up


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    class MockRedis:
        def __init__(self):
            self.data = {}
            self.expiry = {}
        
        async def get(self, key: str) -> str:
            return self.data.get(key)
        
        async def set(self, key: str, value: str, ex: int = None) -> None:
            self.data[key] = value
            if ex:
                self.expiry[key] = ex
        
        async def delete(self, key: str) -> None:
            self.data.pop(key, None)
            self.expiry.pop(key, None)
        
        async def exists(self, key: str) -> bool:
            return key in self.data
        
        async def keys(self, pattern: str = "*") -> list:
            if pattern == "*":
                return list(self.data.keys())
            # Simple pattern matching
            import re
            regex = pattern.replace("*", ".*")
            return [k for k in self.data.keys() if re.match(regex, k)]
        
        async def flushall(self) -> None:
            self.data.clear()
            self.expiry.clear()
    
    return MockRedis()


# ===== Component Mocks =====
@pytest.fixture
def mock_consciousness():
    """Mock consciousness system"""
    mock = Mock()
    mock.get_state = Mock(return_value={
        'awareness_level': 0.8,
        'introspection_depth': 3,
        'metacognitive_cycles': 150,
        'self_model_coherence': 0.92
    })
    mock.introspect = AsyncMock(return_value={
        'thoughts': ['test thought'],
        'insights': ['test insight']
    })
    return mock


@pytest.fixture
def mock_neural_resource_manager():
    """Mock Neural Resource Manager"""
    mock = Mock()
    mock.allocate_resources = AsyncMock(return_value={
        'allocated': True,
        'efficiency_multiplier': 150
    })
    mock.optimize = AsyncMock(return_value={
        'optimization_applied': True,
        'performance_gain': 150.5
    })
    return mock


# ===== WebSocket Mocks =====
@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    ws = Mock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock(return_value='{"type": "test", "data": {}}')
    ws.close = AsyncMock()
    ws.closed = False
    return ws


# ===== File System Fixtures =====
@pytest.fixture
def temp_dir():
    """Provide temporary directory for file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_file_system(temp_dir):
    """Mock file system operations"""
    class MockFileSystem:
        def __init__(self, base_dir):
            self.base_dir = base_dir
        
        def write_file(self, filename: str, content: str) -> Path:
            filepath = self.base_dir / filename
            filepath.write_text(content)
            return filepath
        
        def read_file(self, filename: str) -> str:
            filepath = self.base_dir / filename
            return filepath.read_text()
        
        def exists(self, filename: str) -> bool:
            return (self.base_dir / filename).exists()
    
    return MockFileSystem(temp_dir)


# ===== Test Data Fixtures =====
@pytest.fixture
def sample_conversation():
    """Sample conversation for testing"""
    return {
        'messages': [
            {'role': 'user', 'content': 'Hello JARVIS'},
            {'role': 'assistant', 'content': 'Hello! How can I help you today?'},
            {'role': 'user', 'content': 'What is the weather?'},
        ],
        'context': {
            'user_id': 'test_user_123',
            'session_id': 'test_session_456'
        }
    }


@pytest.fixture
def sample_task():
    """Sample task for testing"""
    return {
        'id': 'task_123',
        'type': 'code_generation',
        'description': 'Generate a Python function to calculate fibonacci',
        'priority': 'high',
        'metadata': {
            'language': 'python',
            'complexity': 'medium'
        }
    }


# ===== Performance Fixtures =====
@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.times = {}
        
        def start(self, name: str):
            self.times[name] = time.time()
        
        def stop(self, name: str) -> float:
            if name not in self.times:
                raise ValueError(f"Timer {name} not started")
            elapsed = time.time() - self.times[name]
            del self.times[name]
            return elapsed
        
        def measure(self, name: str):
            """Context manager for timing"""
            class TimerContext:
                def __init__(self, timer, name):
                    self.timer = timer
                    self.name = name
                    self.elapsed = None
                
                def __enter__(self):
                    self.timer.start(self.name)
                    return self
                
                def __exit__(self, *args):
                    self.elapsed = self.timer.stop(self.name)
            
            return TimerContext(self, name)
    
    return Timer()


# ===== Pytest Markers =====
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async")


# ===== Test Helpers =====
class AsyncContextManager:
    """Helper for testing async context managers"""
    def __init__(self, mock_obj):
        self.mock_obj = mock_obj
    
    async def __aenter__(self):
        return self.mock_obj
    
    async def __aexit__(self, *args):
        pass


def assert_async_called_with(mock: AsyncMock, *args, **kwargs):
    """Helper to assert async mock was called with specific arguments"""
    mock.assert_called_with(*args, **kwargs)


def create_mock_agent(name: str, capabilities: list = None) -> Mock:
    """Create a mock agent for testing"""
    agent = Mock()
    agent.name = name
    agent.capabilities = capabilities or ['general']
    agent.execute = AsyncMock(return_value={'status': 'success'})
    agent.is_available = Mock(return_value=True)
    return agent


# ===== Cleanup =====
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Clean up any resources
    # Reset any global state
    # Clear any caches