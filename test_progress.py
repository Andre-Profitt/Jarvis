#!/usr/bin/env python3
"""
Run tests and show real progress to 100%
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests and show progress"""
    test_dirs = [
        "tests/test_configuration.py",
        "tests/test_database.py", 
        "tests/test_simple_performance_optimizer.py",
        "tests/test_code_generator_agent_complete.py"
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_dirs:
        print(f"\n🧪 Running {test_file}...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        # Parse results
        output = result.stdout + result.stderr
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        skipped = output.count(" SKIPPED")
        
        total_passed += passed
        total_failed += failed
        
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⏭️  Skipped: {skipped}")
    
    # Final summary
    total_tests = total_passed + total_failed
    if total_tests > 0:
        percentage = (total_passed / total_tests) * 100
        print(f"\n📊 TOTAL PROGRESS: {percentage:.1f}%")
        print(f"   ✅ {total_passed} tests passing")
        print(f"   ❌ {total_failed} tests failing")
        
        if percentage == 100:
            print("\n🎉 CONGRATULATIONS! 100% TESTS PASSING! 🎉")
        elif percentage >= 90:
            print("\n🚀 Almost there! Just a few more fixes needed!")
        elif percentage >= 80:
            print("\n💪 Great progress! Keep going!")
    
if __name__ == "__main__":
    run_tests()
