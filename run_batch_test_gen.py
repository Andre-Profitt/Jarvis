#!/usr/bin/env python3
"""Simple batch test generation runner"""
import sys
import os
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM')

# Full paths
base_path = "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/"

# Create a simple coverage report
modules_to_test = [
    {"path": base_path + "core/neural_resource_manager.py", "file": "core/neural_resource_manager.py", "coverage": 0.0, "module_name": "neural_resource_manager", "missing_lines": 100},
    {"path": "core/self_healing_system.py", "file": "core/self_healing_system.py", "coverage": 0.0, "module_name": "self_healing_system", "missing_lines": 100},
    {"path": "core/consciousness_simulation.py", "file": "core/consciousness_simulation.py", "coverage": 0.0, "module_name": "consciousness_simulation", "missing_lines": 100},
    {"path": "core/model_ensemble.py", "file": "core/model_ensemble.py", "coverage": 0.0, "module_name": "model_ensemble", "missing_lines": 100},
    {"path": "core/performance_tracker.py", "file": "core/performance_tracker.py", "coverage": 0.0, "module_name": "performance_tracker", "missing_lines": 100},
    {"path": "core/knowledge_synthesizer.py", "file": "core/knowledge_synthesizer.py", "coverage": 0.0, "module_name": "knowledge_synthesizer", "missing_lines": 100},
    {"path": "core/architecture_evolver.py", "file": "core/architecture_evolver.py", "coverage": 0.0, "module_name": "architecture_evolver", "missing_lines": 100},
    {"path": "core/code_improver.py", "file": "core/code_improver.py", "coverage": 0.0, "module_name": "code_improver", "missing_lines": 100},
    {"path": "core/ast_analyzer.py", "file": "core/ast_analyzer.py", "coverage": 0.0, "module_name": "ast_analyzer", "missing_lines": 100},
    {"path": "core/configuration.py", "file": "core/configuration.py", "coverage": 0.0, "module_name": "configuration", "missing_lines": 100},
]

# Write temporary coverage file
with open('/tmp/coverage.json', 'w') as f:
    json.dump({
        "files": {m["file"]: {"summary": {"percent_covered": m["coverage"]}} for m in modules_to_test}
    }, f)

# Change to temp directory
os.chdir('/tmp')

# Import and run
from tests.batch_test_generator import TestGeneratorBatch

generator = TestGeneratorBatch()
# Manually set modules instead of reading coverage
generator.get_low_coverage_modules = lambda: modules_to_test[:5]

# Process
results = generator.process_batch(max_modules=5, parallel=False)

print("\n✅ Test generation complete!")
print(f"Generated {len(results)} test files")