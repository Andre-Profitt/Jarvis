#!/usr/bin/env python3
"""
Simple verification script for Phase 12 components
"""

import sys
import os
from pathlib import Path
import importlib
import traceback

# Add JARVIS root to path
sys.path.insert(0, str(Path(__file__).parent))

print("\n🔍 JARVIS Phase 12 Component Verification")
print("=" * 50)

# Components to verify
components_to_check = [
    ("Phase 1 - Unified Pipeline", "core.unified_input_pipeline", "UnifiedInputPipeline"),
    ("Phase 2 - Fluid State", "core.fluid_state_management", "FluidStateManager"),
    ("Phase 3 - Neural Resources", "core.neural_resource_manager", "NeuralResourceManager"),
    ("Phase 4 - Self Healing", "core.self_healing_system", "SelfHealingSystem"),
    ("Phase 5 - Quantum Swarm", "core.quantum_swarm_optimization", "QuantumSwarmOptimizer"),
    ("Phase 7 - ML Components", "core.world_class_ml", "JARVISTransformer"),
    ("Phase 8 - Database", "core.database", "DatabaseManager"),
    ("Phase 9 - Monitoring", "core.monitoring", "MonitoringService"),
]

verified_count = 0
total_count = len(components_to_check)

print(f"\nChecking {total_count} core components...\n")

for name, module_path, class_name in components_to_check:
    try:
        # Try to import the module
        module = importlib.import_module(module_path)
        
        # Check if the class exists
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            print(f"✅ {name}: {class_name} found")
            verified_count += 1
        else:
            print(f"❌ {name}: {class_name} not found in module")
            
    except ImportError as e:
        print(f"⚠️  {name}: Import error - {str(e)}")
    except Exception as e:
        print(f"❌ {name}: Error - {str(e)}")

print("\n" + "=" * 50)
print(f"Verification Summary: {verified_count}/{total_count} components verified")

if verified_count == total_count:
    print("✅ All Phase 12 components are available!")
else:
    print(f"⚠️  {total_count - verified_count} components need attention")

# Check for key files
print("\n📁 Checking key Phase 12 files...")
phase12_files = [
    "phase12_integration_testing.py",
    "phase12_deployment_prep.py", 
    "phase12_complete_summary.py",
    "run_phase12.sh"
]

for file_name in phase12_files:
    file_path = Path(__file__).parent / file_name
    if file_path.exists():
        print(f"✅ {file_name} exists")
    else:
        print(f"❌ {file_name} missing")

# Check test results directory
test_results_dir = Path(__file__).parent / "test_results"
if test_results_dir.exists():
    test_files = list(test_results_dir.glob("phase12_*"))
    print(f"\n📊 Found {len(test_files)} test result files")
else:
    print("\n📊 No test results directory found (will be created when tests run)")

print("\n🎯 Phase 12 Verification Complete!")
print("\nTo run full integration tests (requires API keys):")
print("  python3 phase12_integration_testing.py")
print("\nTo prepare for deployment:")
print("  python3 phase12_deployment_prep.py")
