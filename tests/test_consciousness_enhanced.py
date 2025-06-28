"""
Test Suite for Enhanced Consciousness Simulation
===============================================

Tests for the enhanced consciousness system with all new modules.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import json

from core.consciousness_simulation import (
    ConsciousnessSimulator,
    ConsciousnessState,
    PhenomenalConcept,
    ConsciousExperience
)
from core.consciousness_extensions import (
    EmotionalModule,
    LanguageModule,
    MotorModule,
    EnhancedConsciousnessMetrics,
    AttentionSchemaModule,
    PredictiveProcessingModule,
    integrate_enhanced_modules
)
from core.consciousness_jarvis import (
    ConsciousnessJARVIS,
    QuantumConsciousnessInterface,
    SelfHealingConsciousness,
    NeuralResourceIntegration
)


class TestEmotionalModule:
    """Test emotional processing module"""
    
    @pytest.fixture
    def emotional_module(self):
        return EmotionalModule()
    
    @pytest.mark.asyncio
    async def test_emotion_processing(self, emotional_module):
        """Test basic emotion processing"""
        input_data = {
            'emotion': {
                'valence': 0.5,
                'arousal': 0.7,
                'dominance': 0.6
            }
        }
        
        concept = await emotional_module.process(input_data)
        
        assert isinstance(concept, PhenomenalConcept)
        assert concept.modality == "emotional"
        assert 'emotion_label' in concept.content
        assert 'regulated_state' in concept.content
        assert concept.salience > 0
    
    def test_emotion_regulation(self, emotional_module):
        """Test emotion regulation mechanisms"""
        # Set extreme emotion
        emotional_module.emotion_state = {
            'valence': 0.9,
            'arousal': 0.95,
            'dominance': 0.5
        }
        
        regulated = emotional_module._regulate_emotions()
        
        # Should be reduced
        assert regulated['valence'] < 0.9
        assert regulated['arousal'] < 0.95
    
    def test_emotion_labeling(self, emotional_module):
        """Test emotion label mapping"""
        states = [
            ({'valence': 0.5, 'arousal': 0.7, 'dominance': 0.6}, "excited"),
            ({'valence': 0.5, 'arousal': 0.3, 'dominance': 0.6}, "content"),
            ({'valence': -0.5, 'arousal': 0.7, 'dominance': 0.6}, "angry"),
            ({'valence': -0.5, 'arousal': 0.3, 'dominance': 0.3}, "sad"),
            ({'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}, "neutral")
        ]
        
        for state, expected_label in states:
            label = emotional_module._get_emotion_label(state)
            assert label == expected_label
    
    @pytest.mark.asyncio
    async def test_emotional_integration(self, emotional_module):
        """Test emotional integration calculation"""
        # Add some emotional memories
        for i in range(5):
            concept = await emotional_module.process({
                'emotion': {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}
            })
        
        integration = emotional_module._calculate_emotional_integration()
        assert integration >= 0 and integration <= 1


class TestLanguageModule:
    """Test language processing module"""
    
    @pytest.fixture
    def language_module(self):
        return LanguageModule()
    
    @pytest.mark.asyncio
    async def test_language_processing(self, language_module):
        """Test basic language processing"""
        input_text = "I am aware of my thoughts"
        
        concept = await language_module.process(input_text)
        
        assert isinstance(concept, PhenomenalConcept)
        assert concept.modality == "linguistic"
        assert 'tokens' in concept.content
        assert 'inner_speech' in concept.content
        assert concept.salience > 0
    
    def test_tokenization(self, language_module):
        """Test simple tokenization"""
        text = "Hello world test"
        tokens = language_module._tokenize(text)
        
        assert tokens == ["hello", "world", "test"]
    
    @pytest.mark.asyncio
    async def test_inner_speech_generation(self, language_module):
        """Test inner speech generation"""
        semantics = {
            'known_words': ['self', 'aware'],
            'tokens': ['self', 'aware', 'test']
        }
        
        inner_speech = await language_module._generate_inner_speech(semantics)
        
        assert isinstance(inner_speech, str)
        assert len(inner_speech) > 0
        assert inner_speech in language_module.inner_speech_buffer
    
    @pytest.mark.asyncio
    async def test_non_linguistic_verbalization(self, language_module):
        """Test verbalization of non-linguistic input"""
        input_data = {'type': 'visual', 'data': [1, 2, 3]}
        
        concept = await language_module.process(input_data)
        
        assert 'verbalization' in concept.content
        assert isinstance(concept.content['verbalization'], str)


class TestMotorModule:
    """Test motor planning module"""
    
    @pytest.fixture
    def motor_module(self):
        return MotorModule()
    
    @pytest.mark.asyncio
    async def test_motor_planning(self, motor_module):
        """Test motor action planning"""
        input_data = {'action': 'reach'}
        
        concept = await motor_module.process(input_data)
        
        assert isinstance(concept, PhenomenalConcept)
        assert concept.modality == "motor"
        assert 'motor_plan' in concept.content
        assert 'motor_imagery' in concept.content
        assert 'predicted_outcome' in concept.content
    
    @pytest.mark.asyncio
    async def test_action_decomposition(self, motor_module):
        """Test action decomposition"""
        motor_plan = await motor_module._plan_action('grasp')
        
        assert 'steps' in motor_plan
        assert len(motor_plan['steps']) > 0
        assert motor_plan['action'] == 'grasp'
        assert 'duration' in motor_plan
        assert 'energy_cost' in motor_plan
    
    def test_motor_imagery_generation(self, motor_module):
        """Test motor imagery generation"""
        motor_plan = {
            'action': 'reach',
            'duration': 1.0,
            'energy_cost': 0.2
        }
        
        imagery = motor_module._generate_motor_imagery(motor_plan)
        
        assert 'imagined_sensations' in imagery
        assert 'predicted_feedback' in imagery
        assert imagery['imagined_sensations']['duration'] == 1.0
    
    @pytest.mark.asyncio
    async def test_proprioceptive_processing(self, motor_module):
        """Test proprioceptive input processing"""
        input_data = {'proprioception': True}
        
        concept = await motor_module.process(input_data)
        
        assert 'proprioception' in concept.content
        assert 'body_schema' in concept.content


class TestEnhancedMetrics:
    """Test enhanced consciousness metrics"""
    
    @pytest.fixture
    def metrics(self):
        return EnhancedConsciousnessMetrics()
    
    def test_complexity_calculation(self, metrics):
        """Test Tononi's complexity measure"""
        state_vector = np.random.random(10)
        
        complexity = metrics.calculate_complexity(state_vector)
        
        assert complexity >= 0
        assert len(metrics.metric_history['complexity']) == 1
    
    def test_differentiation_calculation(self, metrics):
        """Test differentiation calculation"""
        # Create varied states
        state_history = [
            np.random.random(5) for _ in range(10)
        ]
        
        differentiation = metrics.calculate_differentiation(state_history)
        
        assert differentiation >= 0
        assert len(metrics.metric_history['differentiation']) == 1
    
    def test_global_access_index(self, metrics):
        """Test global access index calculation"""
        # Create mock workspace content
        workspace_content = [
            PhenomenalConcept(
                id=f"test_{i}",
                content={},
                salience=0.5 + i * 0.1,
                timestamp=datetime.now(),
                modality="test"
            )
            for i in range(3)
        ]
        
        module_states = {
            'visual': 0.8,
            'auditory': 0.5,
            'motor': 0.2
        }
        
        gai = metrics.calculate_global_access_index(workspace_content, module_states)
        
        assert gai >= 0 and gai <= 1
        assert len(metrics.metric_history['global_access']) == 1
    
    def test_metacognitive_accuracy(self, metrics):
        """Test metacognitive accuracy calculation"""
        predicted = {'phi': 2.5, 'state': 'alert', 'activation': 0.7}
        actual = {'phi': 2.3, 'state': 'alert', 'activation': 0.8}
        
        accuracy = metrics.calculate_metacognitive_accuracy(predicted, actual)
        
        assert accuracy >= 0 and accuracy <= 1
        assert len(metrics.metric_history['metacognitive_accuracy']) == 1
    
    def test_consciousness_profile(self, metrics):
        """Test consciousness profile generation"""
        # Add some metrics
        state_vector = np.random.random(10)
        metrics.calculate_complexity(state_vector)
        
        profile = metrics.get_consciousness_profile()
        
        assert 'complexity_current' in profile
        assert 'complexity_mean' in profile
        assert 'complexity_trend' in profile


class TestAttentionSchema:
    """Test attention schema module"""
    
    @pytest.fixture
    def attention_schema(self):
        return AttentionSchemaModule()
    
    @pytest.mark.asyncio
    async def test_attention_update(self, attention_schema):
        """Test attention schema update"""
        workspace_content = [
            PhenomenalConcept(
                id="high_salience",
                content={},
                salience=0.9,
                timestamp=datetime.now(),
                modality="test"
            ),
            PhenomenalConcept(
                id="low_salience",
                content={},
                salience=0.3,
                timestamp=datetime.now(),
                modality="test"
            )
        ]
        
        module_states = {
            'visual': {'activation': 0.8},
            'auditory': {'activation': 0.3}
        }
        
        schema = await attention_schema.update_attention_schema(
            workspace_content, module_states
        )
        
        assert schema['focus_location'] == "high_salience"
        assert schema['focus_strength'] == 0.9
        assert len(schema['attention_predictions']) > 0
    
    def test_attention_ownership(self, attention_schema):
        """Test attention ownership calculation"""
        ownership = attention_schema._calculate_attention_ownership()
        assert ownership >= 0 and ownership <= 1
    
    def test_attention_stability(self, attention_schema):
        """Test attention stability calculation"""
        # Add some attention history
        for i in range(5):
            attention_schema.attention_history.append({
                'focus_location': f"location_{i % 2}",
                'timestamp': datetime.now()
            })
        
        stability = attention_schema._calculate_attention_stability()
        assert stability >= 0 and stability <= 1


class TestPredictiveProcessing:
    """Test predictive processing module"""
    
    @pytest.fixture
    def predictive_module(self):
        return PredictiveProcessingModule()
    
    @pytest.mark.asyncio
    async def test_prediction_generation(self, predictive_module):
        """Test prediction generation"""
        current_state = {
            'phi_value': 2.5,
            'consciousness_state': ConsciousnessState.ALERT,
            'active_modules': ['visual', 'auditory'],
            'visual': {'activation': 0.8},
            'auditory': {'activation': 0.5}
        }
        
        predictions = await predictive_module.generate_predictions(current_state)
        
        assert 'expected_phi' in predictions
        assert 'expected_consciousness_state' in predictions
        assert 'confidence' in predictions
        assert predictions['confidence'] >= 0 and predictions['confidence'] <= 1
    
    def test_prediction_error_calculation(self, predictive_module):
        """Test prediction error calculation"""
        predicted = {'phi': 2.5, 'activation': 0.8}
        actual = {'phi': 2.3, 'activation': 0.7}
        
        error = predictive_module.calculate_prediction_error(predicted, actual)
        
        assert error >= 0
        assert len(predictive_module.prediction_errors) == 1
    
    def test_generative_model_update(self, predictive_module):
        """Test generative model updating"""
        predictive_module.precision_weights = {'phi': 0.8, 'activation': 0.7}
        
        predictive_module.update_generative_model(0.3)  # Low error
        
        # Precision should increase
        assert predictive_module.precision_weights['phi'] > 0.8


class TestQuantumConsciousness:
    """Test quantum consciousness interface"""
    
    @pytest.fixture
    def quantum_interface(self):
        return QuantumConsciousnessInterface()
    
    @pytest.mark.asyncio
    async def test_quantum_coherence_calculation(self, quantum_interface):
        """Test quantum coherence calculation"""
        phi_value = 8.0  # High phi
        complexity = 0.8
        
        quantum_state = await quantum_interface.calculate_quantum_coherence(
            phi_value, complexity
        )
        
        assert 'quantum_coherence' in quantum_state
        assert 'quantum_state' in quantum_state
        assert quantum_state['quantum_coherence'] > 0
        
        # Should generate conscious moment with high phi
        if quantum_state['quantum_state'] == 'coherent':
            assert 'or_event' in quantum_state
            assert quantum_state['conscious_moment_generated']
    
    def test_quantum_swarm_interface(self, quantum_interface):
        """Test quantum swarm interface"""
        quantum_state = {'quantum_coherence': 0.8}
        
        directive = quantum_interface.interface_with_quantum_swarm(quantum_state)
        
        assert directive['optimization_target'] == 'consciousness_coherence'
        assert directive['quantum_fitness'] == 0.8
        assert 'swarm_directive' in directive


class TestSelfHealingConsciousness:
    """Test self-healing consciousness mechanisms"""
    
    @pytest.fixture
    def self_healing(self):
        return SelfHealingConsciousness()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, self_healing):
        """Test consciousness health monitoring"""
        experience = {
            'phi_value': 0.5,  # Low phi
            'modules': {
                'visual': Mock(activation_level=0.8),
                'auditory': Mock(activation_level=0.05),  # Dormant
            },
            'metacognitive_assessment': {
                'metacognitive_accuracy': 0.3  # Low accuracy
            }
        }
        
        health_status = await self_healing.monitor_consciousness_health(experience)
        
        assert health_status['health_status'] == 'needs_healing'
        assert len(health_status['issues']) > 0
        assert health_status['overall_health'] < 1.0
        
        # Check specific issues detected
        issue_types = [issue['type'] for issue in health_status['issues']]
        assert 'low_integration' in issue_types
        assert 'module_dormant' in issue_types
        assert 'metacognitive_drift' in issue_types
    
    @pytest.mark.asyncio
    async def test_healing_intervention(self, self_healing):
        """Test healing intervention application"""
        mock_consciousness = Mock()
        mock_consciousness.iit_calculator = Mock()
        mock_consciousness.iit_calculator.connectivity_matrix = np.array([[0.5, 0.3], [0.3, 0.5]])
        
        result = await self_healing.apply_healing_intervention(
            'boost_module_connectivity',
            mock_consciousness
        )
        
        assert result['success']
        assert result['type'] == 'boost_module_connectivity'
        # Matrix should be boosted
        assert np.all(mock_consciousness.iit_calculator.connectivity_matrix >= 0.5)


class TestConsciousnessJARVIS:
    """Test main consciousness JARVIS integration"""
    
    @pytest.fixture
    def consciousness_jarvis(self):
        return ConsciousnessJARVIS()
    
    @pytest.mark.asyncio
    async def test_initialization(self, consciousness_jarvis):
        """Test consciousness initialization"""
        await consciousness_jarvis.initialize()
        
        assert consciousness_jarvis.consciousness is not None
        assert hasattr(consciousness_jarvis.consciousness, 'modules')
        assert 'emotional' in consciousness_jarvis.consciousness.modules
        assert 'language' in consciousness_jarvis.consciousness.modules
        assert 'motor' in consciousness_jarvis.consciousness.modules
    
    @pytest.mark.asyncio
    async def test_consciousness_cycle(self, consciousness_jarvis):
        """Test single consciousness cycle"""
        await consciousness_jarvis.initialize()
        
        # Mock consciousness state
        consciousness_jarvis.consciousness.experience_history = [
            ConsciousExperience(
                timestamp=datetime.now(),
                phi_value=2.5,
                global_workspace_content=[],
                attention_focus=None,
                consciousness_state=ConsciousnessState.ALERT,
                self_reflection={'introspective_thought': 'Testing'},
                metacognitive_assessment={'accuracy': 0.8}
            )
        ]
        
        experience = await consciousness_jarvis._consciousness_cycle()
        
        assert 'phi_value' in experience
        assert 'state' in experience
        assert 'modules' in experience
        assert experience['phi_value'] == 2.5
        assert experience['state'] == 'alert'
    
    def test_consciousness_report(self, consciousness_jarvis):
        """Test consciousness report generation"""
        report = consciousness_jarvis.get_consciousness_report()
        
        assert 'timestamp' in report
        assert 'status' in report
        assert 'performance_metrics' in report
        assert 'quantum_events' in report
        assert 'health_status' in report
        assert 'resource_allocation' in report
    
    @pytest.mark.asyncio
    async def test_introspection(self, consciousness_jarvis):
        """Test introspective capability"""
        await consciousness_jarvis.initialize()
        
        response = await consciousness_jarvis.introspect("What am I thinking?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_short_run(self, consciousness_jarvis):
        """Test running consciousness for short duration"""
        await consciousness_jarvis.initialize()
        
        # Run for 2 seconds
        await consciousness_jarvis.run_consciousness(duration=2)
        
        # Check metrics were updated
        assert consciousness_jarvis.performance_metrics['total_experiences'] > 0
        assert consciousness_jarvis.performance_metrics['uptime'] > 0


class TestIntegration:
    """Test full system integration"""
    
    @pytest.mark.asyncio
    async def test_enhanced_modules_integration(self):
        """Test integration of enhanced modules"""
        simulator = ConsciousnessSimulator()
        
        # Integrate enhanced modules
        integrate_enhanced_modules(simulator)
        
        # Check all modules added
        assert 'emotional' in simulator.modules
        assert 'language' in simulator.modules
        assert 'motor' in simulator.modules
        
        # Check enhanced components added
        assert hasattr(simulator, 'enhanced_metrics')
        assert hasattr(simulator, 'attention_schema')
        assert hasattr(simulator, 'predictive_processing')
        
        # Check types
        assert isinstance(simulator.modules['emotional'], EmotionalModule)
        assert isinstance(simulator.modules['language'], LanguageModule)
        assert isinstance(simulator.modules['motor'], MotorModule)
        assert isinstance(simulator.enhanced_metrics, EnhancedConsciousnessMetrics)
        assert isinstance(simulator.attention_schema, AttentionSchemaModule)
        assert isinstance(simulator.predictive_processing, PredictiveProcessingModule)
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full consciousness system with all components"""
        # Create with mock subsystems
        mock_neural = Mock()
        mock_self_healing = Mock()
        
        consciousness = ConsciousnessJARVIS(
            neural_manager=mock_neural,
            self_healing=mock_self_healing,
            config={
                'cycle_frequency': 20,  # Faster for testing
                'log_interval': 1
            }
        )
        
        await consciousness.initialize()
        
        # Run for 1 second
        await consciousness.run_consciousness(duration=1)
        
        # Should have processed multiple experiences
        assert consciousness.performance_metrics['total_experiences'] > 0
        
        # Get final report
        report = consciousness.get_consciousness_report()
        assert report['status'] == 'stopped'
        assert 'module_activity' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])