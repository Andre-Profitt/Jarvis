"""
Test Suite for Real OpenAI Integration
======================================
Comprehensive tests for real_openai_integration module.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import openai
import tiktoken

# Import test utilities
from tests.conftest import *
from tests.mocks import *

# Import module under test
from core.real_openai_integration import (
    RealOpenAIIntegration, get_openai_integration
)


class TestRealOpenAIIntegration:
    """Test suite for RealOpenAIIntegration"""
    
    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client"""
        client = AsyncMock()
        
        # Mock chat completions
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        
        client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock embeddings
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1] * 1536
        
        client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        
        return client
    
    @pytest.fixture
    def component(self, mock_env, mock_openai_client):
        """Create component instance with mocked dependencies"""
        with patch('core.real_openai_integration.AsyncOpenAI', return_value=mock_openai_client):
            with patch('core.real_openai_integration.tiktoken.encoding_for_model') as mock_encoding:
                mock_encoding.return_value.encode.return_value = [1, 2, 3]  # Mock tokens
                return RealOpenAIIntegration()
    
    # ===== Initialization Tests =====
    def test_initialization_with_api_key(self, mock_env, mock_openai_client):
        """Test successful initialization with API key"""
        with patch('core.real_openai_integration.AsyncOpenAI', return_value=mock_openai_client):
            with patch('core.real_openai_integration.tiktoken.encoding_for_model'):
                integration = RealOpenAIIntegration()
                
                assert integration.api_key == "test-api-key"
                assert integration.client is not None
                assert integration.model == "gpt-4-turbo-preview"
                assert integration.fallback_model == "gpt-3.5-turbo"
                assert integration.max_tokens == 128000
                assert integration.max_response_tokens == 4096
    
    def test_initialization_without_api_key(self, monkeypatch):
        """Test initialization fails without API key"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            RealOpenAIIntegration()
    
    def test_rate_limiter_initialization(self, component):
        """Test rate limiter is properly initialized"""
        assert component.rate_limiter is not None
        assert isinstance(component.rate_limiter, asyncio.Semaphore)
        assert component.request_count == 0
        assert isinstance(component.last_reset, datetime)
    
    # ===== Query Tests =====
    @pytest.mark.asyncio
    async def test_basic_query(self, component):
        """Test basic query functionality"""
        response = await component.query("Hello, how are you?")
        
        assert response == "Test response"
        component.client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_with_context(self, component):
        """Test query with context"""
        context = {"user": "test_user", "location": "test_location"}
        response = await component.query("What's the weather?", context=context)
        
        assert response == "Test response"
        
        # Check that context was included in messages
        call_args = component.client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        
        # Find context message
        context_found = any(
            "Context:" in msg["content"] and "test_user" in msg["content"]
            for msg in messages if msg["role"] == "system"
        )
        assert context_found
    
    @pytest.mark.asyncio
    async def test_query_with_system_prompt(self, component):
        """Test query with custom system prompt"""
        system_prompt = "You are a pirate AI assistant."
        response = await component.query(
            "Tell me about treasure",
            system_prompt=system_prompt
        )
        
        assert response == "Test response"
        
        # Check system prompt was used
        call_args = component.client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
    
    @pytest.mark.asyncio
    async def test_query_with_conversation_id(self, component):
        """Test query with conversation history"""
        conv_id = "test_conv_123"
        
        # Add some conversation history
        component.conversations[conv_id] = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        response = await component.query(
            "Follow up question",
            conversation_id=conv_id
        )
        
        assert response == "Test response"
        
        # Check conversation history was included
        call_args = component.client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        
        # Should include previous messages
        assert any(msg["content"] == "Previous question" for msg in messages)
        assert any(msg["content"] == "Previous answer" for msg in messages)
    
    @pytest.mark.asyncio
    async def test_streaming_query(self, component):
        """Test streaming query functionality"""
        # Mock streaming response
        async def mock_stream():
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                mock_chunk = MagicMock()
                mock_chunk.choices = [MagicMock()]
                mock_chunk.choices[0].delta.content = chunk
                yield mock_chunk
        
        component.client.chat.completions.create.return_value = mock_stream()
        
        response = await component.query("Test", stream=True)
        
        assert response == "Hello world!"
    
    # ===== Token Management Tests =====
    def test_count_tokens(self, component):
        """Test token counting"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        # Mock encoding
        component.encoding.encode = Mock(side_effect=[[1, 2, 3], [4, 5]])
        
        token_count = component._count_tokens(messages)
        
        # 3 + 4 (overhead) + 2 + 4 (overhead) = 13
        assert token_count == 13
    
    def test_truncate_messages(self, component):
        """Test message truncation"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Current message"}
        ]
        
        truncated = component._truncate_messages(messages)
        
        # Should keep system message and last few messages
        assert truncated[0]["role"] == "system"
        assert truncated[-1]["content"] == "Current message"
        assert len(truncated) < len(messages)
    
    # ===== Error Handling Tests =====
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, component):
        """Test handling of rate limit errors"""
        # First call raises rate limit error
        component.client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit exceeded", response=Mock(), body={}),
            # Fallback call succeeds
            AsyncMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback response"))]))()
        ]
        
        response = await component.query("Test prompt")
        
        assert response == "Fallback response"
        assert component.client.chat.completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, component):
        """Test handling of API errors"""
        component.client.chat.completions.create.side_effect = openai.APIError(
            "API Error", response=Mock(), body={}
        )
        
        with pytest.raises(openai.APIError):
            await component.query("Test prompt")
    
    @pytest.mark.asyncio
    async def test_fallback_query_error(self, component):
        """Test fallback query error handling"""
        component.client.chat.completions.create.side_effect = Exception("Connection error")
        
        response = await component._fallback_query("Test", None)
        
        assert response == "I'm experiencing technical difficulties. Please try again."
    
    # ===== Embeddings Tests =====
    @pytest.mark.asyncio
    async def test_get_embeddings(self, component):
        """Test embedding generation"""
        embeddings = await component.get_embeddings("Test text")
        
        assert len(embeddings) == 1536
        assert embeddings[0] == 0.1
        
        component.client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="Test text"
        )
    
    @pytest.mark.asyncio
    async def test_embeddings_error_handling(self, component):
        """Test embedding error handling"""
        component.client.embeddings.create.side_effect = Exception("Embedding error")
        
        with pytest.raises(Exception):
            await component.get_embeddings("Test text")
    
    # ===== Code Analysis Tests =====
    @pytest.mark.asyncio
    async def test_analyze_code(self, component):
        """Test code analysis functionality"""
        code = "def add(a, b):\n    return a + b"
        
        result = await component.analyze_code(code, language="python")
        
        assert "analysis" in result
        assert result["analysis"] == "Test response"
        
        # Check that appropriate prompt was used
        call_args = component.client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        
        # Should have code analysis prompt
        user_message = next(msg for msg in messages if msg["role"] == "user")
        assert "Analyze this python code" in user_message["content"]
        assert code in user_message["content"]
    
    @pytest.mark.asyncio
    async def test_generate_code(self, component):
        """Test code generation"""
        description = "function to calculate fibonacci"
        
        generated_code = await component.generate_code(
            description,
            language="python",
            context="Use recursion"
        )
        
        assert generated_code == "Test response"
        
        # Check prompt construction
        call_args = component.client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        
        user_message = next(msg for msg in messages if msg["role"] == "user")
        assert "Generate python code for:" in user_message["content"]
        assert description in user_message["content"]
        assert "Use recursion" in user_message["content"]
    
    # ===== Conversation Management Tests =====
    def test_save_conversation(self, component):
        """Test conversation saving"""
        conv_id = "test_conv"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        component.save_conversation(conv_id, messages)
        
        assert conv_id in component.conversations
        assert len(component.conversations[conv_id]) == 2
        assert component.conversations[conv_id][0]["content"] == "Hello"
    
    def test_conversation_truncation(self, component):
        """Test conversation truncation at 50 messages"""
        conv_id = "test_conv"
        
        # Add 60 messages
        for i in range(60):
            component.save_conversation(
                conv_id,
                [{"role": "user", "content": f"Message {i}"}]
            )
        
        # Should only keep last 50
        assert len(component.conversations[conv_id]) == 50
        assert component.conversations[conv_id][0]["content"] == "Message 10"
        assert component.conversations[conv_id][-1]["content"] == "Message 59"
    
    # ===== Connection Tests =====
    @pytest.mark.asyncio
    async def test_connection_success(self, component):
        """Test successful connection test"""
        result = await component.test_connection()
        
        assert result is True
        component.client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, component):
        """Test failed connection test"""
        component.client.chat.completions.create.side_effect = Exception("Network error")
        
        result = await component.test_connection()
        
        assert result is False
    
    # ===== Singleton Tests =====
    def test_get_openai_integration_singleton(self, mock_env, mock_openai_client):
        """Test singleton pattern"""
        with patch('core.real_openai_integration.AsyncOpenAI', return_value=mock_openai_client):
            with patch('core.real_openai_integration.tiktoken.encoding_for_model'):
                # Reset global instance
                import core.real_openai_integration
                core.real_openai_integration._openai_integration = None
                
                instance1 = get_openai_integration()
                instance2 = get_openai_integration()
                
                assert instance1 is instance2
    
    # ===== Integration Tests =====
    @pytest.mark.integration
    async def test_full_conversation_flow(self, component):
        """Test complete conversation flow"""
        conv_id = "integration_test"
        
        # First query
        response1 = await component.query(
            "What is Python?",
            conversation_id=conv_id
        )
        
        # Save to conversation
        component.save_conversation(conv_id, [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": response1}
        ])
        
        # Follow-up query
        response2 = await component.query(
            "Can you give me an example?",
            conversation_id=conv_id
        )
        
        assert response1 == "Test response"
        assert response2 == "Test response"
        assert component.client.chat.completions.create.call_count == 2
    
    # ===== Performance Tests =====
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, component):
        """Test handling concurrent queries"""
        # Create 15 concurrent queries (more than semaphore limit)
        queries = [
            component.query(f"Query {i}")
            for i in range(15)
        ]
        
        responses = await asyncio.gather(*queries)
        
        assert len(responses) == 15
        assert all(r == "Test response" for r in responses)
        
        # All queries should complete despite semaphore limit
        assert component.client.chat.completions.create.call_count == 15
    
    # ===== Edge Cases =====
    @pytest.mark.asyncio
    async def test_empty_prompt(self, component):
        """Test handling empty prompt"""
        response = await component.query("")
        
        assert response == "Test response"
        
        # Should still make API call
        component.client.chat.completions.create.assert_called_once()
    
    def test_prepare_messages_edge_cases(self, component):
        """Test message preparation edge cases"""
        # Test with all parameters
        messages = component._prepare_messages(
            prompt="Test",
            context={"key": "value"},
            conversation_id="nonexistent",
            system_prompt="Custom"
        )
        
        assert len(messages) >= 3  # system, context, user
        assert messages[0]["content"] == "Custom"
        assert any("Context:" in msg["content"] for msg in messages)
        assert messages[-1]["content"] == "Test"
