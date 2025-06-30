#!/usr/bin/env python3
"""
Debug the remaining failing tests in Phase 12
"""

import os
import sys
import subprocess
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the test components
from phase12_integration_testing import JARVISIntegrationTester
import asyncio

async def debug_failing_tests():
    """Debug the specific failing tests"""
    
    print("🔍 Debugging Phase 12 Failing Tests\n")
    
    tester = JARVISIntegrationTester()
    
    # Initialize components
    if not await tester.initialize_all_components():
        print("Failed to initialize components")
        return
    
    # Test 1: Pipeline Integration
    print("\n1️⃣ Testing Pipeline Integration...")
    try:
        await tester._test_pipeline_integration()
        print("✅ Pipeline test passed")
    except Exception as e:
        print(f"❌ Pipeline test failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    
    # Test 2: Quantum Swarm
    print("\n2️⃣ Testing Quantum Swarm...")
    try:
        await tester._test_quantum_optimization()
        print("✅ Quantum Swarm test passed")
    except Exception as e:
        print(f"❌ Quantum Swarm test failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    
    # Test 3: End-to-End
    print("\n3️⃣ Testing End-to-End Scenarios...")
    try:
        await tester._test_end_to_end_scenarios()
        print("✅ End-to-End test passed")
    except Exception as e:
        print(f"❌ End-to-End test failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_failing_tests())
