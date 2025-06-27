#!/usr/bin/env python3
"""
Integration tests for JARVIS components
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.real_claude_integration import RealClaudeDesktopIntegration
from core.real_openai_integration import RealOpenAIIntegration
from core.real_elevenlabs_integration import RealElevenLabsIntegration
from core.world_class_swarm import WorldClassSwarmSystem, SwarmTask
from core.world_class_ml import JARVISTransformer, WorldClassTrainer
from core.database import DatabaseManager
from core.websocket_security import WebSocketSecurity
from core.monitoring import MonitoringService, monitor_performance


class TestRealIntegrations:
    """Test real AI integrations"""
    
    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv('OPENAI_API_KEY', 'test_key')
        monkeypatch.setenv('ELEVENLABS_API_KEY', 'test_key')
        monkeypatch.setenv('GEMINI_API_KEY', 'test_key')
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///test_jarvis.db')
    
    @pytest.mark.asyncio
    async def test_claude_integration_setup(self, mock_env):
        """Test Claude Desktop integration setup"""
        claude = RealClaudeDesktopIntegration()
        
        # Test MCP server creation
        await claude.setup_mcp_server()
        
        mcp_file = Path(__file__).parent.parent / "mcp_servers" / "jarvis_mcp.py"
        assert mcp_file.exists()
        
        # Test that MCP server is executable
        assert mcp_file.stat().st_mode & 0o111
    
    @pytest.mark.asyncio
    async def test_openai_integration(self, mock_env):
        """Test OpenAI integration"""
        with patch('openai.AsyncOpenAI') as mock_client:
            # Mock API response
            mock_completion = Mock()
            mock_completion.choices = [Mock(message=Mock(content="Test response"))]
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
            
            openai = RealOpenAIIntegration()
            
            # Test basic query
            response = await openai.query("Test prompt")
            assert response == "Test response"
            
            # Test with context
            response = await openai.query(
                "Test prompt",
                context={"key": "value"},
                temperature=0.5
            )
            assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_elevenlabs_integration(self, mock_env):
        """Test ElevenLabs integration"""
        with patch('elevenlabs.generate') as mock_generate:
            with patch('elevenlabs.play') as mock_play:
                mock_generate.return_value = b"audio_data"
                
                elevenlabs = RealElevenLabsIntegration()
                
                # Test basic speech
                await elevenlabs.speak("Hello world", emotion="happy")
                
                mock_generate.assert_called_once()
                mock_play.assert_called_once()
    
    def test_database_initialization(self, mock_env):
        """Test database setup"""
        db = DatabaseManager()
        
        # Test table creation
        db.create_tables()
        
        # Test conversation creation
        conv_id = db.create_conversation("test_user", {"context": "test"})
        assert conv_id.startswith("conv_")
        
        # Test message addition
        msg_id = db.add_message(conv_id, "user", "Hello")
        assert msg_id.startswith("msg_")
        
        # Test conversation retrieval
        history = db.get_conversation_history(conv_id)
        assert len(history) == 1
        assert history[0]['content'] == "Hello"
    
    def test_websocket_security(self):
        """Test WebSocket security implementation"""
        security = WebSocketSecurity()
        
        # Test token generation
        device_info = {
            "device_id": "test_device",
            "device_type": "test",
            "device_name": "Test Device"
        }
        
        token = security.generate_device_token(device_info)
        assert isinstance(token, str)
        
        # Test token verification
        payload = security.verify_token(token)
        assert payload is not None
        assert payload['device_id'] == "test_device"
        
        # Test encryption/decryption
        message = {"test": "data"}
        encrypted = security.encrypt_message(message)
        decrypted = security.decrypt_message(encrypted)
        assert decrypted == message


class TestWorldClassML:
    """Test world-class ML components"""
    
    def test_jarvis_transformer_creation(self):
        """Test transformer model creation"""
        model = JARVISTransformer(
            vocab_size=1000,
            embed_dim=512,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=1024,
            use_flash_attn=False  # Disable for testing
        )
        
        # Test model structure
        assert model.vocab_size == 1000
        assert model.embed_dim == 512
        assert len(model.blocks) == 4
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (2, 10, 1000)
    
    def test_rotary_embeddings(self):
        """Test RoPE implementation"""
        from core.world_class_ml import RotaryPositionalEmbedding
        
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(None, seq_len=100)
        
        assert cos.shape == (100, 32)  # dim/2
        assert sin.shape == (100, 32)
    
    def test_swiglu_activation(self):
        """Test SwiGLU activation"""
        from core.world_class_ml import SwiGLU
        
        swiglu = SwiGLU(dim=256)
        x = torch.randn(2, 10, 256)
        output = swiglu(x)
        
        assert output.shape == x.shape
    
    @pytest.mark.asyncio
    async def test_curriculum_learning(self):
        """Test curriculum learning"""
        from core.world_class_ml import CurriculumLearning
        
        curriculum = CurriculumLearning(difficulties=[0.3, 0.5, 0.7, 1.0])
        
        # Test difficulty progression
        assert curriculum.get_batch_difficulty(0, 10) == 0.0
        assert curriculum.get_batch_difficulty(5, 10) == 0.5
        assert curriculum.get_batch_difficulty(10, 10) == 1.0
        
        # Test data filtering
        data = [
            {'difficulty': 0.2},
            {'difficulty': 0.6},
            {'difficulty': 0.9}
        ]
        
        filtered = curriculum.filter_data(data, 0.5)
        assert len(filtered) == 1
        assert filtered[0]['difficulty'] == 0.2


class TestSwarmIntelligence:
    """Test swarm intelligence components"""
    
    @pytest.mark.asyncio
    async def test_swarm_agent_creation(self):
        """Test swarm agent creation"""
        # Initialize Ray for testing
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        swarm = WorldClassSwarmSystem()
        
        # Create test agent
        agent = await swarm.create_agent(
            agent_type="test",
            capabilities={"testing", "debugging"}
        )
        
        assert "test_0" in swarm.agents
        assert swarm.swarm_graph.has_node("test_0")
    
    @pytest.mark.asyncio
    async def test_contract_net_protocol(self):
        """Test contract net task allocation"""
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        swarm = WorldClassSwarmSystem()
        
        # Create agents
        await swarm.create_agent("worker", {"execution", "analysis"})
        await swarm.create_agent("worker", {"execution", "coding"})
        
        # Submit task
        task = SwarmTask(
            task_id="test_task",
            task_type="analysis",
            payload={"analyze": "test"},
            required_capabilities={"execution", "analysis"},
            reward=10.0
        )
        
        contractor = await swarm.submit_task(task)
        assert contractor is not None
    
    def test_acl_message_creation(self):
        """Test FIPA-ACL message structure"""
        from core.world_class_swarm import ACLMessage
        
        msg = ACLMessage(
            performative=ACLMessage.Performative.INFORM,
            sender="agent1",
            receiver="agent2",
            content={"info": "test"}
        )
        
        assert msg.performative == ACLMessage.Performative.INFORM
        assert msg.sender == "agent1"
        assert msg.content == {"info": "test"}
        assert msg.conversation_id is not None


class TestMonitoring:
    """Test monitoring and observability"""
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        from core.monitoring import MetricsCollector, MetricEvent
        
        collector = MetricsCollector()
        
        # Collect system metrics
        metrics = collector.collect_system_metrics()
        
        assert 'cpu' in metrics
        assert 'memory' in metrics
        assert 'disk' in metrics
        assert 'network' in metrics
    
    @pytest.mark.asyncio
    async def test_health_checks(self):
        """Test health check system"""
        from core.monitoring import HealthChecker
        
        checker = HealthChecker()
        
        # Register test check
        def test_check():
            return {'status': 'healthy', 'message': 'Test OK'}
        
        checker.register_check('test', test_check)
        
        # Run checks
        results = await checker.run_checks()
        
        assert results['status'] == 'healthy'
        assert 'test' in results['checks']
        assert results['checks']['test']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring decorator"""
        
        @monitor_performance("test_component")
        async def test_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_function()
        assert result == "success"


class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, mock_env):
        """Test complete conversation flow"""
        # Initialize components
        db = DatabaseManager()
        
        # Create conversation
        conv_id = db.create_conversation("test_user")
        
        # Add user message
        db.add_message(conv_id, "user", "Hello JARVIS")
        
        # Mock AI response
        with patch('core.real_openai_integration.openai_integration.query') as mock_query:
            mock_query.return_value = "Hello! How can I help you today?"
            
            # Add assistant response
            db.add_message(
                conv_id, 
                "assistant", 
                "Hello! How can I help you today?",
                model_used="gpt4",
                confidence=0.95
            )
        
        # Record learning
        learning_id = db.record_learning(
            "preference",
            {"user_prefers": "friendly_greeting"},
            conversation_id=conv_id,
            confidence=0.8
        )
        
        # Verify conversation
        history = db.get_conversation_history(conv_id)
        assert len(history) == 2
        
        # Verify learning
        learnings = db.get_learnings("preference")
        assert len(learnings) > 0
        assert learnings[0]['content']['user_prefers'] == "friendly_greeting"


# Performance benchmarks

class TestPerformance:
    """Performance benchmarks"""
    
    def test_transformer_inference_speed(self):
        """Benchmark transformer inference"""
        model = JARVISTransformer(
            vocab_size=50000,
            embed_dim=1024,
            num_layers=12,
            num_heads=16,
            num_kv_heads=4,
            use_flash_attn=False
        )
        
        model.eval()
        
        # Prepare input
        batch_size = 1
        seq_len = 512
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        
        # Benchmark
        import time
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        
        elapsed = time.time() - start
        avg_time = elapsed / 10
        
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        assert avg_time < 0.5  # Should be under 500ms
    
    def test_database_query_performance(self, mock_env):
        """Benchmark database operations"""
        db = DatabaseManager()
        
        # Create test data
        conv_id = db.create_conversation("perf_test")
        
        import time
        start = time.time()
        
        # Add 100 messages
        for i in range(100):
            db.add_message(conv_id, "user", f"Message {i}")
        
        # Query history
        history = db.get_conversation_history(conv_id)
        
        elapsed = time.time() - start
        
        print(f"100 messages + query: {elapsed*1000:.2f}ms")
        assert elapsed < 1.0  # Should be under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])