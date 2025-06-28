"""
Tests for LLM Research Integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

from core.llm_research_integration import (
    LLMProvider,
    LLMResponse,
    ClaudeCLI,
    GeminiCLI,
    DualLLM,
    SourcePlugin,
    ArxivPlugin,
    SemanticScholarPlugin,
    LLMEnhancedResearcher
)
from core.llm_research_jarvis import (
    ResearchTask,
    LLMResearchJARVIS
)
from core.llm_research_quickstart import QuickResearch


class TestLLMResponse:
    """Test LLM response functionality"""
    
    def test_response_creation(self):
        """Test creating LLM response"""
        response = LLMResponse(
            provider=LLMProvider.CLAUDE,
            content="Test response",
            confidence=0.9,
            reasoning="Test reasoning"
        )
        
        assert response.provider == LLMProvider.CLAUDE
        assert response.content == "Test response"
        assert response.confidence == 0.9
        assert response.reasoning == "Test reasoning"
        assert isinstance(response.timestamp, datetime)
        
    def test_response_with_sources(self):
        """Test response with sources"""
        response = LLMResponse(
            provider=LLMProvider.GEMINI,
            content="Response with sources",
            confidence=0.85,
            sources_cited=["source1", "source2"]
        )
        
        assert len(response.sources_cited) == 2
        assert "source1" in response.sources_cited


@pytest.mark.asyncio
class TestClaudeCLI:
    """Test Claude CLI interface"""
    
    async def test_claude_analyze(self):
        """Test Claude analysis"""
        claude = ClaudeCLI()
        
        # Mock subprocess run
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"analysis": "Test analysis", "confidence": 0.9}'
            )
            
            response = await claude.analyze("Test prompt")
            
            assert isinstance(response, LLMResponse)
            assert response.provider == LLMProvider.CLAUDE
            assert "analysis" in response.content
            
    async def test_claude_synthesize(self):
        """Test Claude synthesis"""
        claude = ClaudeCLI()
        
        sources = [
            {"title": "Paper 1", "abstract": "Abstract 1"},
            {"title": "Paper 2", "abstract": "Abstract 2"}
        ]
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"synthesis": "Combined insights", "key_points": []}'
            )
            
            response = await claude.synthesize(sources, "Synthesize these papers")
            
            assert isinstance(response, LLMResponse)
            assert "synthesis" in response.content


@pytest.mark.asyncio
class TestGeminiCLI:
    """Test Gemini CLI interface"""
    
    async def test_gemini_analyze(self):
        """Test Gemini analysis"""
        gemini = GeminiCLI()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='Analysis result from Gemini'
            )
            
            response = await gemini.analyze("Test prompt")
            
            assert isinstance(response, LLMResponse)
            assert response.provider == LLMProvider.GEMINI
            
    async def test_gemini_validate(self):
        """Test Gemini validation"""
        gemini = GeminiCLI()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"is_valid": true, "confidence": 0.95}'
            )
            
            response = await gemini.validate(
                "Test claim",
                ["evidence 1", "evidence 2"]
            )
            
            assert isinstance(response, LLMResponse)
            assert "is_valid" in response.content


@pytest.mark.asyncio
class TestDualLLM:
    """Test dual LLM functionality"""
    
    async def test_dual_analyze(self):
        """Test dual LLM analysis"""
        claude = AsyncMock()
        gemini = AsyncMock()
        
        claude.analyze.return_value = LLMResponse(
            provider=LLMProvider.CLAUDE,
            content='{"result": "Claude result"}',
            confidence=0.9
        )
        
        gemini.analyze.return_value = LLMResponse(
            provider=LLMProvider.GEMINI,
            content='{"result": "Gemini result"}',
            confidence=0.85
        )
        
        dual = DualLLM(claude, gemini)
        response = await dual.analyze("Test prompt")
        
        assert response.provider == LLMProvider.BOTH
        data = json.loads(response.content)
        assert "claude" in data
        assert "gemini" in data
        assert "consensus" in data
        assert "disagreements" in data


@pytest.mark.asyncio
class TestArxivPlugin:
    """Test ArXiv plugin"""
    
    async def test_arxiv_search(self):
        """Test ArXiv search"""
        plugin = ArxivPlugin()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = """
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Test Paper</title>
                    <summary>Test abstract</summary>
                    <author><name>Test Author</name></author>
                    <link href="http://arxiv.org/abs/1234.5678"/>
                    <published>2024-01-01T00:00:00Z</published>
                </entry>
            </feed>
            """
            mock_get.return_value.__aenter__.return_value = mock_response
            
            sources = await plugin.search("quantum computing", max_results=5)
            
            assert len(sources) > 0
            assert sources[0]['title'] == "Test Paper"
            assert sources[0]['abstract'] == "Test abstract"


class TestResearchTask:
    """Test research task"""
    
    def test_task_creation(self):
        """Test creating research task"""
        task = ResearchTask(
            id="test123",
            topic="AI Safety",
            type="deep",
            depth="comprehensive"
        )
        
        assert task.id == "test123"
        assert task.topic == "AI Safety"
        assert task.type == "deep"
        assert task.depth == "comprehensive"
        assert task.status == "pending"
        
    def test_task_with_questions(self):
        """Test task with custom questions"""
        questions = ["Q1", "Q2", "Q3"]
        task = ResearchTask(
            id="test456",
            topic="Quantum ML",
            custom_questions=questions
        )
        
        assert len(task.custom_questions) == 3
        assert "Q1" in task.custom_questions


@pytest.mark.asyncio
class TestLLMResearchJARVIS:
    """Test JARVIS integration"""
    
    async def test_initialization(self):
        """Test JARVIS research initialization"""
        jarvis = LLMResearchJARVIS()
        
        with patch('core.llm_research_jarvis.ClaudeCLI'), \
             patch('core.llm_research_jarvis.GeminiCLI'), \
             patch('core.llm_research_jarvis.WorldClassSwarmSystem'):
            
            await jarvis.initialize()
            
            assert jarvis._initialized
            assert jarvis.researcher is not None
            
    async def test_research_caching(self):
        """Test research result caching"""
        jarvis = LLMResearchJARVIS()
        jarvis._initialized = True
        
        # Mock researcher
        jarvis.researcher = AsyncMock()
        jarvis.researcher.research_with_llm.return_value = {
            'topic': 'Test Topic',
            'key_findings': [{'finding': 'Test finding', 'confidence': 0.9}],
            'summary': 'Test summary'
        }
        
        # Mock neural allocation
        with patch('core.llm_research_jarvis.neural_jarvis') as mock_neural:
            mock_neural.neural_manager.allocate_resources.return_value = {
                'allocated_resources': {}
            }
            mock_neural.neural_manager.release_resources.return_value = None
            
            # First call
            result1 = await jarvis.research("Test Topic")
            
            # Second call (should use cache)
            result2 = await jarvis.research("Test Topic")
            
            # Should only call researcher once due to caching
            jarvis.researcher.research_with_llm.assert_called_once()
            
            # Results should be identical
            assert result1 == result2
            
    async def test_research_types(self):
        """Test different research types"""
        jarvis = LLMResearchJARVIS()
        jarvis._initialized = True
        
        # Test deep research
        task = ResearchTask(
            id="deep1",
            topic="Deep Learning",
            type="deep",
            depth="comprehensive"
        )
        
        # Test comparative research
        comparative_task = ResearchTask(
            id="comp1",
            topic="ML Frameworks",
            type="comparative",
            context={'topics': ['TensorFlow', 'PyTorch', 'JAX']}
        )
        
        # Test hypothesis research
        hypothesis_task = ResearchTask(
            id="hyp1",
            topic="AGI Development",
            type="hypothesis"
        )
        
        # Verify task types are handled differently
        assert task.type == "deep"
        assert comparative_task.type == "comparative"
        assert hypothesis_task.type == "hypothesis"


@pytest.mark.asyncio
class TestQuickResearch:
    """Test quick research interface"""
    
    async def test_quick_setup(self):
        """Test quick research setup"""
        qr = QuickResearch()
        
        with patch('core.llm_research_jarvis.llm_research_jarvis.initialize'):
            await qr.setup()
            assert qr.initialized
            
    async def test_quick_research(self):
        """Test quick research function"""
        qr = QuickResearch()
        qr.initialized = True
        
        with patch('core.llm_research_jarvis.llm_research_jarvis.research') as mock_research:
            mock_research.return_value = {
                'topic': 'Test',
                'summary': 'Test summary',
                'key_findings': [
                    {'finding': 'Finding 1', 'confidence': 0.9},
                    {'finding': 'Finding 2', 'confidence': 0.8}
                ]
            }
            
            result = await qr.research("Test topic")
            
            assert 'summary' in result
            assert 'key_findings' in result
            assert len(result['key_findings']) == 2


def test_llm_provider_enum():
    """Test LLM provider enum"""
    assert LLMProvider.CLAUDE.value == "claude"
    assert LLMProvider.GEMINI.value == "gemini"
    assert LLMProvider.BOTH.value == "both"


@pytest.mark.asyncio
async def test_end_to_end_research():
    """Test end-to-end research flow"""
    
    # This would be an integration test with mocked external services
    with patch('subprocess.run') as mock_run, \
         patch('aiohttp.ClientSession.get') as mock_get:
        
        # Mock CLI responses
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "Mocked LLM response"}'
        )
        
        # Mock API responses
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'data': [{'title': 'Paper', 'abstract': 'Abstract'}]
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Initialize system
        jarvis = LLMResearchJARVIS()
        await jarvis.initialize()
        
        # Conduct research
        result = await jarvis.research(
            "Test research topic",
            depth="quick"
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'jarvis_context' in result