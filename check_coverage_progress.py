#!/usr/bin/env python3
"""Check current test coverage progress"""

import subprocess
import sys
from pathlib import Path
import json

print("🚀 Checking Test Coverage Progress...")
print("=" * 60)

# Run coverage on specific test files we've implemented
implemented_tests = [
    "test_self_healing_system.py",
    "test_monitoring.py", 
    "test_advanced_integration.py",
    "test_consciousness_simulation.py",
    "test_model_ensemble.py",
    "test_performance_tracker.py"
]

# Run each test individually to see coverage
total_tests = 0
passing_tests = 0

for test_file in implemented_tests:
    test_path = f"/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/tests/{test_file}"
    if Path(test_path).exists():
        print(f"\n📋 Running {test_file}...")
        
        # Extract module name from test file
        module_name = test_file.replace("test_", "").replace(".py", "")
        
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_path,
            f"--cov=core.{module_name}",
            "--cov-report=term-missing",
            "-q",
            "--tb=no"
        ], capture_output=True, text=True)
        
        # Count tests
        if "passed" in result.stdout:
            # Extract test count
            import re
            match = re.search(r'(\d+) passed', result.stdout)
            if match:
                test_count = int(match.group(1))
                total_tests += test_count
                passing_tests += test_count
                print(f"✅ {test_count} tests passed")
        
        # Extract coverage for the specific module
        lines = result.stdout.split('\n')
        for line in lines:
            if f"core/{module_name}.py" in line and "%" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "%" in part:
                        coverage = part.strip('%')
                        print(f"📊 Module coverage: {coverage}%")
                        break

print(f"\n" + "=" * 60)
print(f"📈 Summary:")
print(f"Total tests run: {total_tests}")
print(f"Tests passing: {passing_tests}")
print(f"Success rate: {100 * passing_tests / total_tests if total_tests > 0 else 0:.1f}%")

# Count total test files
test_dir = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/tests")
all_test_files = list(test_dir.glob("test_*.py"))
all_test_files = [f for f in all_test_files if "template" not in f.name and "example" not in f.name]

print(f"\nTest files created: {len(all_test_files)}")
print(f"Test files with implementation: {len(implemented_tests)}")
print(f"Test files needing implementation: {len(all_test_files) - len(implemented_tests)}")

print("\n🎯 Next Steps:")
print("1. Implement test logic for remaining generated test files")
print("2. Focus on high-value modules (database, config_manager, etc.)")
print("3. Run full coverage report to measure progress")