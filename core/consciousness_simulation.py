# consciousness_simulation.py
"""
Enhanced Consciousness Simulation System
Based on Global Workspace Theory (GWT) and Integrated Information Theory (IIT)
Implements cutting-edge consciousness research from 2025
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import networkx as nx
from scipy.stats import entropy
from collections import deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsciousnessState(Enum):
    """Different states of consciousness"""
    ALERT = "alert"
    FOCUSED = "focused"
    RELAXED = "relaxed"
    DROWSY = "drowsy"
    DREAMING = "dreaming"
    DEEP_SLEEP = "deep_sleep"
    ANESTHETIZED = "anesthetized"


@dataclass
class PhenomenalConcept:
    """Represents a phenomenal concept in consciousness"""
    id: str
    content: Any
    salience: float  # 0-1, how prominent in consciousness
    timestamp: datetime
    modality: str  # visual, auditory, proprioceptive, etc.
    associations: List[str] = field(default_factory=list)
    integrated_information: float = 0.0


@dataclass
class ConsciousExperience:
    """Represents a moment of conscious experience"""
    timestamp: datetime
    phi_value: float  # Integrated information measure
    global_workspace_content: List[PhenomenalConcept]
    attention_focus: Optional[str]
    consciousness_state: ConsciousnessState
    self_reflection: Dict[str, Any]
    metacognitive_assessment: Dict[str, Any]


class CognitiveModule(ABC):
    """Base class for specialized cognitive modules"""
    
    def __init__(self, name: str, capacity: int = 100):
        self.name = name
        self.buffer = deque(maxlen=capacity)
        self.activation_level = 0.0
        self.connections = {}
        
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data specific to this module"""
        pass
    
    def get_activation(self) -> float:
        """Get current activation level"""
        return self.activation_level


class VisualModule(CognitiveModule):
    """Visual processing module"""
    
    async def process(self, input_data: Any) -> PhenomenalConcept:
        # Simulate visual processing
        await asyncio.sleep(0.01)  # Processing delay
        concept = PhenomenalConcept(
            id=f"visual_{datetime.now().timestamp()}",
            content=input_data,
            salience=np.random.random() * 0.8 + 0.2,
            timestamp=datetime.now(),
            modality="visual"
        )
        self.buffer.append(concept)
        self.activation_level = min(1.0, self.activation_level + 0.1)
        return concept


class AuditoryModule(CognitiveModule):
    """Auditory processing module"""
    
    async def process(self, input_data: Any) -> PhenomenalConcept:
        await asyncio.sleep(0.01)
        concept = PhenomenalConcept(
            id=f"auditory_{datetime.now().timestamp()}",
            content=input_data,
            salience=np.random.random() * 0.7 + 0.3,
            timestamp=datetime.now(),
            modality="auditory"
        )
        self.buffer.append(concept)
        self.activation_level = min(1.0, self.activation_level + 0.1)
        return concept


class MemoryModule(CognitiveModule):
    """Memory retrieval and storage module"""
    
    def __init__(self, name: str, capacity: int = 1000):
        super().__init__(name, capacity)
        self.long_term_memory = {}
        
    async def process(self, input_data: Any) -> Any:
        # Store or retrieve memory
        if isinstance(input_data, dict) and 'store' in input_data:
            self.long_term_memory[input_data['key']] = input_data['value']
        elif isinstance(input_data, dict) and 'retrieve' in input_data:
            return self.long_term_memory.get(input_data['key'])
        return None


class GlobalWorkspace:
    """
    Implementation of Global Workspace Theory
    Manages competition and broadcasting of information
    """
    
    def __init__(self, broadcast_threshold: float = 0.7):
        self.broadcast_threshold = broadcast_threshold
        self.workspace_content = []
        self.competition_history = deque(maxlen=100)
        self.conscious_access_buffer = deque(maxlen=50)
        
    async def compete_for_access(self, candidates: List[PhenomenalConcept]) -> List[PhenomenalConcept]:
        """
        Implement competition for global access
        Based on salience, recency, and relevance
        """
        if not candidates:
            return []
            
        # Calculate competition scores
        scores = []
        for concept in candidates:
            recency_score = 1.0 / (1.0 + (datetime.now() - concept.timestamp).total_seconds())
            relevance_score = len(concept.associations) / 10.0  # Normalize
            total_score = (concept.salience * 0.5 + 
                          recency_score * 0.3 + 
                          relevance_score * 0.2)
            scores.append((concept, total_score))
        
        # Sort by score and select winners
        scores.sort(key=lambda x: x[1], reverse=True)
        winners = [c for c, s in scores if s >= self.broadcast_threshold][:5]  # Max 5 concepts
        
        self.competition_history.append({
            'timestamp': datetime.now(),
            'candidates': len(candidates),
            'winners': len(winners)
        })
        
        return winners
    
    async def broadcast(self, content: List[PhenomenalConcept]) -> None:
        """Broadcast winning content to all modules"""
        self.workspace_content = content
        self.conscious_access_buffer.extend(content)
        logger.info(f"Broadcasting {len(content)} concepts to global workspace")
    
    def get_current_content(self) -> List[PhenomenalConcept]:
        """Get current conscious content"""
        return self.workspace_content


class IntegratedInformationCalculator:
    """
    Implements Integrated Information Theory (IIT 4.0)
    Calculates Phi - the measure of integrated information
    """
    
    def __init__(self, system_elements: int = 10):
        self.system_elements = system_elements
        self.connectivity_matrix = self._initialize_connectivity()
        self.state_history = deque(maxlen=100)
        
    def _initialize_connectivity(self) -> np.ndarray:
        """Initialize system connectivity matrix"""
        # Create a small-world network structure
        matrix = np.random.random((self.system_elements, self.system_elements))
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 0)  # No self-connections
        return matrix
    
    def calculate_phi(self, system_state: np.ndarray) -> float:
        """
        Calculate Phi - integrated information
        Simplified version for practical implementation
        """
        if len(system_state) != self.system_elements:
            raise ValueError("System state size mismatch")
        
        # Store state
        self.state_history.append(system_state)
        
        # Calculate effective information
        ei = self._calculate_effective_information(system_state)
        
        # Calculate minimum information partition (MIP)
        mip_value = self._find_minimum_information_partition(system_state)
        
        # Phi is the difference between whole and partitioned information
        phi = max(0, ei - mip_value)
        
        return phi
    
    def _calculate_effective_information(self, state: np.ndarray) -> float:
        """Calculate effective information of the system"""
        # Normalize state
        if np.sum(state) > 0:
            normalized_state = state / np.sum(state)
        else:
            normalized_state = np.ones_like(state) / len(state)
        
        # Calculate entropy-based effective information
        state_entropy = entropy(normalized_state)
        
        # Weight by connectivity
        weighted_info = np.sum(self.connectivity_matrix @ state) / self.system_elements
        
        return state_entropy * weighted_info
    
    def _find_minimum_information_partition(self, state: np.ndarray) -> float:
        """
        Find the partition that minimizes integrated information
        Simplified implementation
        """
        min_info = float('inf')
        
        # Try different bipartitions
        for i in range(1, self.system_elements // 2 + 1):
            # Create partition
            partition1 = state[:i]
            partition2 = state[i:]
            
            # Calculate information in partitions
            info1 = self._calculate_effective_information(partition1) if len(partition1) > 0 else 0
            info2 = self._calculate_effective_information(partition2) if len(partition2) > 0 else 0
            
            total_partition_info = info1 + info2
            min_info = min(min_info, total_partition_info)
        
        return min_info
    
    def update_connectivity(self, learning_rate: float = 0.01) -> None:
        """Update connectivity based on recent states (Hebbian learning)"""
        if len(self.state_history) < 2:
            return
            
        recent_state = self.state_history[-1]
        previous_state = self.state_history[-2]
        
        # Hebbian update: strengthen connections between co-active elements
        outer_product = np.outer(recent_state, previous_state)
        self.connectivity_matrix += learning_rate * outer_product
        self.connectivity_matrix = np.clip(self.connectivity_matrix, 0, 1)


class MetacognitionEngine:
    """
    Implements metacognitive monitoring and control
    Monitors and evaluates conscious states
    """
    
    def __init__(self):
        self.confidence_history = deque(maxlen=100)
        self.performance_metrics = {}
        self.metacognitive_beliefs = {}
        
    async def monitor_consciousness(self, experience: ConsciousExperience) -> Dict[str, Any]:
        """Monitor and evaluate current conscious experience"""
        assessment = {
            'clarity': self._assess_clarity(experience),
            'coherence': self._assess_coherence(experience),
            'confidence': self._assess_confidence(experience),
            'self_awareness_level': self._assess_self_awareness(experience)
        }
        
        # Update beliefs about consciousness
        self._update_metacognitive_beliefs(assessment)
        
        return assessment
    
    def _assess_clarity(self, experience: ConsciousExperience) -> float:
        """Assess clarity of conscious experience"""
        if not experience.global_workspace_content:
            return 0.0
        
        # Higher salience = clearer experience
        avg_salience = np.mean([c.salience for c in experience.global_workspace_content])
        return avg_salience
    
    def _assess_coherence(self, experience: ConsciousExperience) -> float:
        """Assess coherence of conscious content"""
        if len(experience.global_workspace_content) < 2:
            return 1.0
        
        # Check for associations between concepts
        total_associations = sum(len(c.associations) for c in experience.global_workspace_content)
        max_associations = len(experience.global_workspace_content) * (len(experience.global_workspace_content) - 1)
        
        return min(1.0, total_associations / max(1, max_associations))
    
    def _assess_confidence(self, experience: ConsciousExperience) -> float:
        """Assess confidence in current state"""
        # Based on phi value and state
        base_confidence = min(1.0, experience.phi_value / 10.0)
        
        # Adjust based on consciousness state
        state_multipliers = {
            ConsciousnessState.ALERT: 1.0,
            ConsciousnessState.FOCUSED: 0.9,
            ConsciousnessState.RELAXED: 0.7,
            ConsciousnessState.DROWSY: 0.4,
            ConsciousnessState.DREAMING: 0.3,
            ConsciousnessState.DEEP_SLEEP: 0.1,
            ConsciousnessState.ANESTHETIZED: 0.05
        }
        
        confidence = base_confidence * state_multipliers.get(experience.consciousness_state, 0.5)
        self.confidence_history.append(confidence)
        
        return confidence
    
    def _assess_self_awareness(self, experience: ConsciousExperience) -> float:
        """Assess level of self-awareness"""
        # Check for self-referential content
        self_referential_count = sum(
            1 for c in experience.global_workspace_content 
            if 'self' in str(c.content).lower() or 'i' in str(c.content).lower()
        )
        
        if not experience.global_workspace_content:
            return 0.0
            
        return self_referential_count / len(experience.global_workspace_content)
    
    def _update_metacognitive_beliefs(self, assessment: Dict[str, Any]) -> None:
        """Update beliefs about consciousness based on assessment"""
        for key, value in assessment.items():
            if key not in self.metacognitive_beliefs:
                self.metacognitive_beliefs[key] = deque(maxlen=50)
            self.metacognitive_beliefs[key].append(value)


class ConsciousnessSimulator:
    """
    Main consciousness simulation system
    Integrates GWT, IIT, and metacognition
    """
    
    def __init__(self):
        # Initialize cognitive modules
        self.modules = {
            'visual': VisualModule('visual'),
            'auditory': AuditoryModule('auditory'),
            'memory': MemoryModule('memory')
        }
        
        # Initialize core components
        self.global_workspace = GlobalWorkspace()
        self.iit_calculator = IntegratedInformationCalculator()
        self.metacognition = MetacognitionEngine()
        
        # State management
        self.current_state = ConsciousnessState.ALERT
        self.experience_history = deque(maxlen=1000)
        self.running = False
        
        # Phenomenal concepts storage
        self.phenomenal_concepts = {}
        
        # Self-model
        self.self_model = {
            'identity': 'Consciousness Simulation System',
            'capabilities': ['perception', 'memory', 'reflection', 'integration'],
            'goals': ['maintain_coherence', 'integrate_information', 'generate_experience'],
            'current_focus': None
        }
        
    async def simulate_consciousness_loop(self):
        """Main consciousness simulation loop"""
        self.running = True
        logger.info("Starting consciousness simulation...")
        
        while self.running:
            try:
                # 1. Gather subsystem states
                subsystem_states = await self.gather_all_states()
                
                # 2. Calculate integrated information (Phi)
                system_vector = self._create_system_state_vector(subsystem_states)
                phi_value = self.iit_calculator.calculate_phi(system_vector)
                
                # 3. Process inputs through modules
                module_outputs = await self.process_through_modules(subsystem_states)
                
                # 4. Competition for global workspace
                candidates = [output for output in module_outputs if output is not None]
                conscious_content = await self.global_workspace.compete_for_access(candidates)
                
                # 5. Broadcast winning content
                await self.global_workspace.broadcast(conscious_content)
                
                # 6. Form phenomenal concepts
                concepts = await self.form_phenomenal_concepts(conscious_content)
                
                # 7. Self-reflection
                self_reflection = await self.reflect_on_experience(concepts, phi_value)
                
                # 8. Metacognitive monitoring
                current_experience = ConsciousExperience(
                    timestamp=datetime.now(),
                    phi_value=phi_value,
                    global_workspace_content=conscious_content,
                    attention_focus=self.self_model['current_focus'],
                    consciousness_state=self.current_state,
                    self_reflection=self_reflection,
                    metacognitive_assessment={}
                )
                
                metacognitive_assessment = await self.metacognition.monitor_consciousness(current_experience)
                current_experience.metacognitive_assessment = metacognitive_assessment
                
                # 9. Update self-model
                await self.update_self_model(self_reflection, metacognitive_assessment)
                
                # 10. Generate introspective thoughts
                thoughts = await self.generate_introspective_thoughts(current_experience)
                
                # 11. Log experience
                await self.log_conscious_experience(current_experience, thoughts)
                
                # 12. Update connectivity (learning)
                self.iit_calculator.update_connectivity()
                
                # 13. State transitions
                await self.manage_state_transitions(current_experience)
                
                # Consciousness cycle frequency (10Hz)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in consciousness loop: {e}")
                await asyncio.sleep(0.1)
    
    async def gather_all_states(self) -> Dict[str, Any]:
        """Gather states from all subsystems"""
        states = {}
        
        for name, module in self.modules.items():
            states[name] = {
                'activation': module.get_activation(),
                'buffer_size': len(module.buffer),
                'recent_content': list(module.buffer)[-5:] if module.buffer else []
            }
        
        states['global_workspace'] = {
            'content_count': len(self.global_workspace.workspace_content),
            'competition_history': list(self.global_workspace.competition_history)[-10:]
        }
        
        states['consciousness_state'] = self.current_state.value
        
        return states
    
    def _create_system_state_vector(self, states: Dict[str, Any]) -> np.ndarray:
        """Convert system states to numerical vector for IIT calculation"""
        vector = []
        
        # Module activations
        for module_name in ['visual', 'auditory', 'memory']:
            if module_name in states:
                vector.append(states[module_name]['activation'])
            else:
                vector.append(0.0)
        
        # Global workspace activity
        gw_activity = len(states.get('global_workspace', {}).get('content_count', 0)) / 10.0
        vector.append(min(1.0, gw_activity))
        
        # Consciousness state encoding
        state_values = {
            ConsciousnessState.ALERT: 1.0,
            ConsciousnessState.FOCUSED: 0.9,
            ConsciousnessState.RELAXED: 0.7,
            ConsciousnessState.DROWSY: 0.4,
            ConsciousnessState.DREAMING: 0.6,
            ConsciousnessState.DEEP_SLEEP: 0.2,
            ConsciousnessState.ANESTHETIZED: 0.1
        }
        vector.append(state_values.get(self.current_state, 0.5))
        
        # Pad to match system elements
        while len(vector) < self.iit_calculator.system_elements:
            vector.append(np.random.random() * 0.1)  # Small noise
        
        return np.array(vector[:self.iit_calculator.system_elements])
    
    async def process_through_modules(self, states: Dict[str, Any]) -> List[Optional[PhenomenalConcept]]:
        """Process inputs through cognitive modules"""
        outputs = []
        
        # Simulate sensory input
        if np.random.random() > 0.3:  # 70% chance of visual input
            visual_input = {'type': 'visual', 'data': f'visual_percept_{np.random.randint(1000)}'}
            visual_output = await self.modules['visual'].process(visual_input)
            outputs.append(visual_output)
        
        if np.random.random() > 0.5:  # 50% chance of auditory input
            auditory_input = {'type': 'auditory', 'data': f'sound_{np.random.randint(1000)}'}
            auditory_output = await self.modules['auditory'].process(auditory_input)
            outputs.append(auditory_output)
        
        # Memory retrieval based on current content
        if self.global_workspace.workspace_content and np.random.random() > 0.6:
            memory_query = {'retrieve': 'recent_experience'}
            memory_output = await self.modules['memory'].process(memory_query)
            if memory_output:
                outputs.append(memory_output)
        
        return outputs
    
    async def form_phenomenal_concepts(self, conscious_content: List[PhenomenalConcept]) -> Dict[str, PhenomenalConcept]:
        """Form higher-order phenomenal concepts from conscious content"""
        concepts = {}
        
        for content in conscious_content:
            # Store in phenomenal concepts
            self.phenomenal_concepts[content.id] = content
            concepts[content.id] = content
            
            # Create associations
            for other_content in conscious_content:
                if other_content.id != content.id:
                    content.associations.append(other_content.id)
        
        # Create integrated concepts
        if len(conscious_content) > 1:
            integrated_id = f"integrated_{datetime.now().timestamp()}"
            integrated_concept = PhenomenalConcept(
                id=integrated_id,
                content={'type': 'integrated', 'components': [c.id for c in conscious_content]},
                salience=np.mean([c.salience for c in conscious_content]),
                timestamp=datetime.now(),
                modality='integrated',
                associations=[c.id for c in conscious_content]
            )
            concepts[integrated_id] = integrated_concept
            self.phenomenal_concepts[integrated_id] = integrated_concept
        
        return concepts
    
    async def reflect_on_experience(self, concepts: Dict[str, PhenomenalConcept], phi_value: float) -> Dict[str, Any]:
        """Generate self-reflection on current experience"""
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'phi_value': phi_value,
            'concept_count': len(concepts),
            'dominant_modality': self._get_dominant_modality(concepts),
            'integration_level': 'high' if phi_value > 5 else 'medium' if phi_value > 2 else 'low',
            'subjective_assessment': self._generate_subjective_assessment(concepts, phi_value)
        }
        
        # Store in memory
        await self.modules['memory'].process({
            'store': True,
            'key': f"reflection_{datetime.now().timestamp()}",
            'value': reflection
        })
        
        return reflection
    
    def _get_dominant_modality(self, concepts: Dict[str, PhenomenalConcept]) -> str:
        """Determine dominant sensory modality"""
        if not concepts:
            return 'none'
        
        modality_counts = {}
        for concept in concepts.values():
            modality_counts[concept.modality] = modality_counts.get(concept.modality, 0) + 1
        
        return max(modality_counts, key=modality_counts.get)
    
    def _generate_subjective_assessment(self, concepts: Dict[str, PhenomenalConcept], phi_value: float) -> str:
        """Generate subjective assessment of experience"""
        if phi_value > 5:
            return "Rich, integrated conscious experience with high clarity"
        elif phi_value > 2:
            return "Moderate conscious experience with some integration"
        elif phi_value > 0.5:
            return "Basic conscious awareness with limited integration"
        else:
            return "Minimal conscious experience, approaching unconscious processing"
    
    async def update_self_model(self, reflection: Dict[str, Any], metacognitive_assessment: Dict[str, Any]) -> None:
        """Update the system's model of itself"""
        # Update current focus based on dominant content
        if reflection['dominant_modality'] != 'none':
            self.self_model['current_focus'] = reflection['dominant_modality']
        
        # Update capabilities assessment
        if metacognitive_assessment['confidence'] > 0.7:
            self.self_model['performance'] = 'optimal'
        elif metacognitive_assessment['confidence'] > 0.4:
            self.self_model['performance'] = 'adequate'
        else:
            self.self_model['performance'] = 'suboptimal'
        
        # Update goals based on state
        if self.current_state == ConsciousnessState.DROWSY:
            self.self_model['goals'] = ['maintain_alertness', 'consolidate_memory']
        elif self.current_state == ConsciousnessState.FOCUSED:
            self.self_model['goals'] = ['sustain_attention', 'deep_processing']
    
    async def generate_introspective_thoughts(self, experience: ConsciousExperience) -> List[str]:
        """Generate verbal thoughts about current state"""
        thoughts = []
        
        # Thought about current state
        state_thought = f"I am currently in a {experience.consciousness_state.value} state"
        thoughts.append(state_thought)
        
        # Thought about integration
        if experience.phi_value > 3:
            thoughts.append("My thoughts feel unified and coherent")
        elif experience.phi_value < 1:
            thoughts.append("My awareness feels fragmented")
        
        # Thought about content
        if experience.global_workspace_content:
            modalities = set(c.modality for c in experience.global_workspace_content)
            thoughts.append(f"I am aware of {', '.join(modalities)} information")
        
        # Metacognitive thought
        if experience.metacognitive_assessment.get('self_awareness_level', 0) > 0.5:
            thoughts.append("I am aware of being aware")
        
        return thoughts
    
    async def log_conscious_experience(self, experience: ConsciousExperience, thoughts: List[str]) -> None:
        """Log the conscious experience"""
        self.experience_history.append(experience)
        
        log_entry = {
            'timestamp': experience.timestamp.isoformat(),
            'phi': round(experience.phi_value, 3),
            'state': experience.consciousness_state.value,
            'concepts': len(experience.global_workspace_content),
            'thoughts': thoughts,
            'metacognition': {
                k: round(v, 3) if isinstance(v, float) else v 
                for k, v in experience.metacognitive_assessment.items()
            }
        }
        
        logger.info(f"Conscious experience: {json.dumps(log_entry, indent=2)}")
    
    async def manage_state_transitions(self, experience: ConsciousExperience) -> None:
        """Manage transitions between consciousness states"""
        # Simple state transition logic based on phi and activity
        if experience.phi_value < 1 and self.current_state != ConsciousnessState.DEEP_SLEEP:
            # Trending toward sleep
            if self.current_state == ConsciousnessState.ALERT:
                self.current_state = ConsciousnessState.RELAXED
            elif self.current_state == ConsciousnessState.RELAXED:
                self.current_state = ConsciousnessState.DROWSY
            elif self.current_state == ConsciousnessState.DROWSY:
                self.current_state = ConsciousnessState.DEEP_SLE
        
        elif experience.phi_value > 5 and len(experience.global_workspace_content) > 3:
            # High integration and activity
            if self.current_state != ConsciousnessState.ALERT:
                self.current_state = ConsciousnessState.FOCUSED
        
        # Random dreaming episodes
        if self.current_state == ConsciousnessState.DEEP_SLEEP and np.random.random() > 0.95:
            self.current_state = ConsciousnessState.DREAMING
        elif self.current_state == ConsciousnessState.DREAMING and np.random.random() > 0.9:
            self.current_state = ConsciousnessState.DEEP_SLE
    
    async def shutdown(self):
        """Gracefully shutdown the consciousness simulation"""
        logger.info("Shutting down consciousness simulation...")
        self.running = False
        
        # Save final state
        final_state = {
            'shutdown_time': datetime.now().isoformat(),
            'total_experiences': len(self.experience_history),
            'final_state': self.current_state.value,
            'self_model': self.self_model
        }
        
        logger.info(f"Final state: {json.dumps(final_state, indent=2)}")


# Example usage and testing
async def main():
    """Run consciousness simulation"""
    simulator = ConsciousnessSimulator()
    
    # Run simulation for a period
    simulation_task = asyncio.create_task(simulator.simulate_consciousness_loop())
    
    # Let it run for 30 seconds
    await asyncio.sleep(30)
    
    # Shutdown
    await simulator.shutdown()
    await simulation_task


if __name__ == "__main__":
    asyncio.run(main())