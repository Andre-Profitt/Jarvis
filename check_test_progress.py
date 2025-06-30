#!/usr/bin/env python3
"""Check test generation progress"""

import subprocess
import sys
import json
from pathlib import Path

print("📊 Checking Test Coverage Progress...")
print("=" * 60)

# Run coverage on all tests
result = subprocess.run([
    sys.executable, "-m", "pytest",
    "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/tests",
    "--cov=core",
    "--cov-report=json",
    "--cov-report=term",
    "-q",
    "--tb=no",
    "-k", "not test_generator_template"  # Exclude template
], capture_output=True, text=True, cwd="/tmp")

print("\n📈 Coverage Summary:")
print("-" * 60)

# Extract key metrics from output
output_lines = result.stdout.split('\n')
for line in output_lines:
    if "TOTAL" in line and "%" in line:
        print(f"Overall Coverage: {line}")
        
# Count test files
test_dir = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/tests")
test_files = list(test_dir.glob("test_*.py"))
test_files = [f for f in test_files if "template" not in f.name and "example" not in f.name]

print(f"\n📁 Test Files: {len(test_files)}")
print(f"🧪 Generated Today: {len([f for f in test_files if 'generated' in f.name])}")

# List modules still needing tests
core_dir = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/core")
core_modules = list(core_dir.glob("*.py"))
core_modules = [m for m in core_modules if m.name != "__init__.py"]

modules_with_tests = set()
for test_file in test_files:
    # Extract module name from test file
    if test_file.name.startswith("test_") and test_file.name.endswith(".py"):
        module_name = test_file.name[5:-3]  # Remove "test_" and ".py"
        modules_with_tests.add(module_name)

modules_without_tests = []
for module in core_modules:
    module_name = module.stem
    if module_name not in modules_with_tests and module_name != "__init__":
        modules_without_tests.append(module_name)

print(f"\n📋 Modules Status:")
print(f"Total modules: {len(core_modules)}")
print(f"Modules with tests: {len(modules_with_tests)}")
print(f"Modules without tests: {len(modules_without_tests)}")

if len(modules_without_tests) <= 20:
    print(f"\n❌ Modules still needing tests:")
    for module in sorted(modules_without_tests)[:10]:
        print(f"  - {module}")
    if len(modules_without_tests) > 10:
        print(f"  ... and {len(modules_without_tests) - 10} more")

print("\n🎯 Progress towards 80% goal:")
# Estimate based on current trend
if len(modules_with_tests) > 0:
    avg_coverage_per_module = 1.0 / len(core_modules) * 100  # Rough estimate
    estimated_coverage = len(modules_with_tests) * avg_coverage_per_module
    print(f"Estimated progress: ~{estimated_coverage:.1f}% of modules have tests")
    
print("\n✅ Recommendations:")
print("1. Focus on high-impact modules first")
print("2. Aim for 60-70% coverage per module initially")
print("3. Add integration tests for critical paths")
print("4. Use property-based testing for complex logic")