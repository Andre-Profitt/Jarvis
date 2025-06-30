#!/usr/bin/env python3
"""Find core modules without corresponding test files."""

import os
from pathlib import Path

# Get all Python files in core directory (excluding __init__.py and subdirectories)
core_dir = Path('core')
test_dir = Path('tests')

core_files = set()
for f in core_dir.glob('*.py'):
    if f.name != '__init__.py':
        core_files.add(f.stem)

# Get all test files
test_files = set()
for f in test_dir.glob('test_*.py'):
    # Remove 'test_' prefix to get the module name
    module_name = f.stem[5:]  # Remove 'test_' prefix
    test_files.add(module_name)

# Find core modules without tests
missing_tests = sorted(core_files - test_files)

print('Core modules without corresponding test files:')
print('=' * 60)
print('\nIMPORTANT MODULES (agents, managers, integrators, etc.):')
print('-' * 60)

important_keywords = ['agent', 'manager', 'integrator', 'integration', 'engine', 
                     'optimizer', 'processor', 'factory', 'controller', 'handler',
                     'coordinator', 'orchestrator', 'system', 'core']

important_missing = []
other_missing = []

for module in missing_tests:
    is_important = any(keyword in module.lower() for keyword in important_keywords)
    if is_important:
        important_missing.append(module)
    else:
        other_missing.append(module)

for module in important_missing:
    print(f'  - core/{module}.py')

print(f'\nOTHER MODULES:')
print('-' * 60)
for module in other_missing:
    print(f'  - core/{module}.py')

print(f'\nSUMMARY:')
print('=' * 60)
print(f'Total modules without tests: {len(missing_tests)}')
print(f'Important modules without tests: {len(important_missing)}')
print(f'Total core modules: {len(core_files)}')
print(f'Test coverage: {len(test_files.intersection(core_files))}/{len(core_files)} ({len(test_files.intersection(core_files))/len(core_files)*100:.1f}%)')

# Show which test files exist but don't have corresponding core modules
extra_tests = sorted(test_files - core_files)
if extra_tests:
    print(f'\nTest files without corresponding core modules:')
    print('-' * 60)
    for test in extra_tests:
        print(f'  - tests/test_{test}.py')