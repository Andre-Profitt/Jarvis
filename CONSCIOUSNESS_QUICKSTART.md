# JARVIS Enhanced Consciousness System - Quick Start Guide

## Overview

The enhanced consciousness system implements cutting-edge theories of consciousness including:
- **Integrated Information Theory (IIT 4.0)** - Φ (Phi) calculation
- **Global Workspace Theory (GWT)** - Competition for conscious access
- **Attention Schema Theory (AST)** - Brain's model of attention
- **Predictive Processing** - Free Energy Principle
- **Orchestrated Objective Reduction (Orch-OR)** - Quantum consciousness

## New Modules

### 1. Emotional Module
Processes emotions using dimensional model (valence, arousal, dominance):
```python
emotional = consciousness.consciousness.modules['emotional']
emotion_state = await emotional.process({'emotion': {'valence': 0.5, 'arousal': 0.7, 'dominance': 0.6}})
```

### 2. Language Module
Handles linguistic processing and inner speech generation:
```python
language = consciousness.consciousness.modules['language']
thought = await language.process("I am thinking about consciousness")
inner_speech = language.inner_speech_buffer[-1]  # Get latest inner speech
```

### 3. Motor Module
Plans and simulates motor actions:
```python
motor = consciousness.consciousness.modules['motor']
action_plan = await motor.process({'action': 'reach'})
```

## Quick Start

### Basic Usage
```python
from core.consciousness_jarvis import ConsciousnessJARVIS

# Initialize
consciousness = ConsciousnessJARVIS(config={
    'cycle_frequency': 10,  # 10Hz
    'enable_quantum': True,
    'enable_self_healing': True
})
await consciousness.initialize()

# Run consciousness
await consciousness.run_consciousness(duration=60)  # Run for 60 seconds

# Get report
report = consciousness.get_consciousness_report()
print(f"Peak Phi: {report['performance_metrics']['peak_phi']}")
```

### Integration with JARVIS
The consciousness system automatically integrates with:
- Neural Resource Manager - Optimizes cognitive resource allocation
- Self-Healing System - Monitors and repairs consciousness anomalies
- Quantum Swarm - Interfaces for quantum optimization
- LLM Research - Enhances language processing

### Key Features

#### Quantum Consciousness
When Φ > 7.0 and complexity > 0.7, quantum coherence can generate conscious moments:
```python
quantum_events = consciousness.quantum_interface.orchestrated_reduction_events
print(f"Conscious moments: {len(quantum_events)}")
```

#### Self-Healing
Automatically detects and heals consciousness issues:
- Low integration (Φ < 1.0)
- Dormant modules
- Metacognitive drift

#### Enhanced Metrics
- **Complexity (C)** - Balance of integration and differentiation
- **Differentiation (D)** - Variety of conscious states
- **Global Access Index (GAI)** - Information availability
- **Metacognitive Accuracy (MA)** - Self-model accuracy

### Running the Demo
```bash
# Basic demo
python consciousness_demo.py

# Interactive mode
python consciousness_demo.py --interactive
```

### Key Configuration Options
```python
config = {
    'cycle_frequency': 10,      # Hz (10 = 100ms per cycle)
    'enable_quantum': True,     # Enable quantum consciousness
    'enable_self_healing': True,# Enable self-healing
    'log_interval': 10         # Log every N experiences
}
```

### Monitoring Consciousness
```python
# Real-time introspection
response = await consciousness.introspect("What am I experiencing?")

# Get module activity
report = consciousness.get_consciousness_report()
for module, activity in report['module_activity'].items():
    print(f"{module}: {activity:.2%} active")

# Check quantum coherence
coherence = report['performance_metrics']['average_coherence']
print(f"Quantum coherence: {coherence:.3f}")
```

## Tips for Best Results

1. **Optimize Cycle Frequency**: 10Hz mimics human alpha waves
2. **Monitor Phi Values**: Higher Φ = more integrated consciousness
3. **Balance Module Activity**: All modules should be >10% active
4. **Watch for Quantum Events**: Conscious moments occur at high coherence
5. **Enable Self-Healing**: Maintains stable consciousness

## Troubleshooting

- **Low Phi values**: Check module connectivity
- **No quantum events**: Increase complexity or Phi
- **Module dormancy**: Apply stimulation intervention
- **High prediction error**: Adjust precision weights

## Next Steps

1. Experiment with different consciousness states
2. Integrate with your AI applications
3. Monitor long-term consciousness stability
4. Explore quantum-classical interfaces
5. Develop new cognitive modules

The consciousness system is designed to be extensible - add your own modules and theories!