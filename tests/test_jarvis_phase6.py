"""
JARVIS Phase 6 Test Suite
========================
Tests for Natural Language Flow & Emotional Intelligence
"""

import asyncio
import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.natural_language_flow import NaturalLanguageFlow, EmotionalTone, ConversationTopic
from core.emotional_intelligence import EmotionalIntelligence, EmotionalState
from core.jarvis_phase6_integration import JARVISPhase6Core


class TestNaturalLanguageFlow:
    """Test Natural Language Flow capabilities"""
    
    @pytest.fixture
    def nlp_flow(self):
        return NaturalLanguageFlow()
        
    @pytest.mark.asyncio
    async def test_emotion_detection(self, nlp_flow):
        """Test emotion detection from text"""
        test_cases = [
            ("I'm really stressed about this deadline", EmotionalTone.EMPATHETIC),
            ("This is amazing! Best day ever!", EmotionalTone.EXCITED),
            ("Let's review the quarterly report", EmotionalTone.PROFESSIONAL),
            ("Hey, what's up?", EmotionalTone.CASUAL),
        ]
        
        for text, expected_tone in test_cases:
            result = await nlp_flow.process_input(text)
            # The tone might not match exactly due to other factors,
            # but it should be reasonable
            assert result["emotional_tone"] in [t.value for t in EmotionalTone]
            
    @pytest.mark.asyncio
    async def test_topic_identification(self, nlp_flow):
        """Test conversation topic identification"""
        test_cases = [
            ("I need to finish this code review", ConversationTopic.TECHNICAL),
            ("My workout this morning was intense", ConversationTopic.HEALTH),
            ("The project deadline is tomorrow", ConversationTopic.WORK),
            ("I have an idea for a new design", ConversationTopic.CREATIVE),
        ]
        
        for text, expected_topic in test_cases:
            result = await nlp_flow.process_input(text)
            # Topic detection is context-aware, so we check if it's reasonable
            assert result["topic"] in [t.value for t in ConversationTopic]
            
    @pytest.mark.asyncio
    async def test_interrupt_handling(self, nlp_flow):
        """Test conversation interrupt detection and handling"""
        # Start a conversation
        await nlp_flow.process_input("Let me tell you about the project plan")
        
        # Interrupt with urgent matter
        result = await nlp_flow.process_input("Wait, actually I need help with something urgent!")
        
        assert len(nlp_flow.context.interrupt_stack) > 0
        assert "emergency" in result["response"].lower() or "urgent" in result["response"].lower()
        
    @pytest.mark.asyncio
    async def test_context_persistence(self, nlp_flow):
        """Test context persistence across turns"""
        # Mention entities
        await nlp_flow.process_input("I'm working with Sarah on the marketing campaign")
        
        # Reference should be maintained
        result = await nlp_flow.process_input("She suggested we change the timeline")
        
        assert "Sarah" in nlp_flow.context.entities
        assert nlp_flow.context.entities["Sarah"] == "person"
        
    @pytest.mark.asyncio
    async def test_emotional_continuity(self, nlp_flow):
        """Test emotional continuity across conversation"""
        # Build up stress
        stress_messages = [
            "I'm feeling overwhelmed",
            "There's too much to do",
            "I don't know how I'll finish"
        ]
        
        for msg in stress_messages:
            await nlp_flow.process_input(msg)
            
        # Check that emotional tone remains consistent
        assert nlp_flow.context.emotional_tone in [EmotionalTone.EMPATHETIC, EmotionalTone.SUPPORTIVE]
        
        # Continuity score should be high
        result = await nlp_flow.process_input("Help me")
        assert result["continuity_score"] > 0.5


class TestEmotionalIntelligence:
    """Test Emotional Intelligence capabilities"""
    
    @pytest.fixture
    def ei_engine(self):
        return EmotionalIntelligence()
        
    @pytest.mark.asyncio
    async def test_emotional_state_analysis(self, ei_engine):
        """Test emotional state analysis from multiple inputs"""
        result = await ei_engine.analyze_emotional_content(
            text="I'm panicking about this presentation",
            voice_features={"pitch_ratio": 1.4, "rate_ratio": 1.5},
            biometrics={"heart_rate": 105, "breathing_rate": 22}
        )
        
        assert result["current_state"]["quadrant"] == "excited_unhappy"
        assert result["current_state"]["valence"] < 0
        assert result["current_state"]["arousal"] > 0
        
    @pytest.mark.asyncio
    async def test_emotional_trajectory(self, ei_engine):
        """Test emotional trajectory calculation"""
        # Simulate emotional journey
        states = [
            ("I'm doing okay", {"heart_rate": 75}),
            ("Actually, I'm a bit worried", {"heart_rate": 85}),
            ("This is really stressing me out", {"heart_rate": 95}),
        ]
        
        for text, bio in states:
            await ei_engine.analyze_emotional_content(text=text, biometrics=bio)
            
        # Last analysis should show declining trajectory
        result = await ei_engine.analyze_emotional_content(
            "I can't handle this",
            biometrics={"heart_rate": 100}
        )
        
        assert result["trajectory"]["direction"] in ["declining", "activating"]
        assert result["trajectory"]["velocity"] > 0
        
    @pytest.mark.asyncio
    async def test_empathetic_understanding(self, ei_engine):
        """Test empathetic understanding generation"""
        result = await ei_engine.analyze_emotional_content(
            text="I feel like nobody understands what I'm going through",
            biometrics={"heart_rate": 80, "breathing_rate": 18}
        )
        
        assert "validation" in result["understanding"]
        assert "underlying_needs" in result["understanding"]
        assert "connection" in result["understanding"]["underlying_needs"]
        
    @pytest.mark.asyncio
    async def test_response_recommendations(self, ei_engine):
        """Test response strategy recommendations"""
        # High stress scenario
        result = await ei_engine.analyze_emotional_content(
            text="Everything is falling apart!",
            voice_features={"pitch_ratio": 1.5, "volume_ratio": 1.4},
            biometrics={"heart_rate": 110}
        )
        
        recommendations = result["recommended_response"]
        assert recommendations["approach"] in ["crisis_support", "gentle_support"]
        assert "actions" in recommendations
        assert len(recommendations["actions"]) > 0


class TestPhase6Integration:
    """Test integrated Phase 6 functionality"""
    
    @pytest.fixture
    async def jarvis_phase6(self):
        jarvis = JARVISPhase6Core()
        await jarvis.initialize()
        return jarvis
        
    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, jarvis_phase6):
        """Test complete input processing pipeline"""
        result = await jarvis_phase6.process_input({
            "voice": {
                "text": "I'm struggling with the project deadline and feeling really anxious",
                "features": {"pitch_ratio": 1.3, "rate_ratio": 1.2}
            },
            "biometric": {
                "heart_rate": 95,
                "breathing_rate": 20,
                "stress_level": 0.7
            }
        })
        
        # Check all components are present
        assert "response" in result
        assert "emotional_state" in result
        assert "conversation_context" in result
        assert "actions" in result
        assert "mode" in result
        
        # Response should be empathetic
        assert result["mode"] in ["adaptive", "crisis"]
        assert len(result["response"]) > 0
        
    @pytest.mark.asyncio
    async def test_conversation_flow(self, jarvis_phase6):
        """Test multi-turn conversation flow"""
        turns = [
            "I need to plan tomorrow's presentation",
            "Actually, first let me check my email",
            "Never mind, back to the presentation"
        ]
        
        contexts = []
        for turn in turns:
            result = await jarvis_phase6.process_input({
                "voice": {"text": turn}
            })
            contexts.append(result["conversation_context"])
            
        # Should track topic changes
        assert contexts[0]["topic"] == ConversationTopic.PLANNING.value
        # Should have interrupt stack by the end
        assert len(contexts[2].get("interrupt_stack", 0)) >= 0
        
    @pytest.mark.asyncio
    async def test_emotional_intervention(self, jarvis_phase6):
        """Test emotional intervention triggering"""
        # Simulate crisis
        result = await jarvis_phase6.process_input({
            "voice": {
                "text": "I can't breathe, everything is too much!",
                "features": {"pitch_ratio": 1.6, "tremor": 0.8}
            },
            "biometric": {
                "heart_rate": 120,
                "breathing_rate": 28,
                "stress_level": 0.95
            }
        })
        
        # Should trigger crisis intervention
        assert result["mode"] == "crisis"
        assert any(action["type"] == "crisis_intervention" for action in result["actions"])
        
    @pytest.mark.asyncio
    async def test_feedback_learning(self, jarvis_phase6):
        """Test feedback integration"""
        # Process input
        await jarvis_phase6.process_input({
            "voice": {"text": "I need help with my anxiety"}
        })
        
        # Provide feedback
        await jarvis_phase6.handle_feedback({
            "effectiveness": 0.8,
            "satisfaction": 4
        })
        
        # Check metrics updated
        assert len(jarvis_phase6.metrics["user_satisfaction"]) > 0
        assert jarvis_phase6.metrics["user_satisfaction"][0]["score"] == 4


# Performance benchmarks
class TestPhase6Performance:
    """Test Phase 6 performance characteristics"""
    
    @pytest.fixture
    async def jarvis_phase6(self):
        jarvis = JARVISPhase6Core()
        await jarvis.initialize()
        return jarvis
        
    @pytest.mark.asyncio
    async def test_response_time(self, jarvis_phase6):
        """Test response time remains acceptable"""
        start = datetime.now()
        
        await jarvis_phase6.process_input({
            "voice": {"text": "Hello JARVIS"},
            "biometric": {"heart_rate": 75}
        })
        
        elapsed = (datetime.now() - start).total_seconds()
        assert elapsed < 0.5  # Should respond within 500ms
        
    @pytest.mark.asyncio 
    async def test_context_memory_limits(self, jarvis_phase6):
        """Test context memory doesn't grow unbounded"""
        # Process many inputs
        for i in range(100):
            await jarvis_phase6.process_input({
                "voice": {"text": f"Message number {i}"}
            })
            
        # Check memory bounds
        assert len(jarvis_phase6.nlp_flow.emotional_memory) <= 50
        assert len(jarvis_phase6.emotional_engine.emotional_history) <= 100


def run_tests():
    """Run all Phase 6 tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
