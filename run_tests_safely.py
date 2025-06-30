#!/usr/bin/env python3
"""
Safe test runner that handles missing API keys
"""

import os
import sys
import subprocess

def run_tests_with_dummy_keys():
    """Run tests with dummy API keys to avoid initialization errors"""
    
    # Set dummy API keys if not present
    env = os.environ.copy()
    
    if 'OPENAI_API_KEY' not in env:
        env['OPENAI_API_KEY'] = 'dummy-key-for-testing'
        print("ℹ️  Using dummy OpenAI API key for testing")
    
    if 'ANTHROPIC_API_KEY' not in env:
        env['ANTHROPIC_API_KEY'] = 'dummy-key-for-testing'
        print("ℹ️  Using dummy Anthropic API key for testing")
    
    if 'ELEVENLABS_API_KEY' not in env:
        env['ELEVENLABS_API_KEY'] = 'dummy-key-for-testing'
        print("ℹ️  Using dummy ElevenLabs API key for testing")
    
    print("\n🧪 Running Phase 12 Integration Tests...\n")
    
    # Run the actual test
    result = subprocess.run(
        [sys.executable, 'phase12_integration_testing.py'],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests_with_dummy_keys()
    sys.exit(exit_code)
