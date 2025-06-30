#\!/usr/bin/env python3
"""
Run all tests and generate comprehensive report
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_test_suite():
    """Run complete test suite with detailed reporting"""
    
    # Test categories
    test_suites = {
        "Configuration": "tests/test_configuration.py",
        "Database": "tests/test_database.py", 
        "Performance Optimizer": "tests/test_simple_performance_optimizer.py",
        "Code Generator": "tests/test_code_generator_agent_complete.py",
        "All Core Tests": "tests/test_core.py",
        "Integrations": "tests/test_integrations.py"
    }
    
    results = {}
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    print("🚀 JARVIS Test Suite Runner")
    print("=" * 60)
    
    for suite_name, test_file in test_suites.items():
        if not Path(test_file).exists():
            print(f"\n⚠️  {suite_name}: Test file not found")
            continue
            
        print(f"\n🧪 Running {suite_name} tests...")
        
        # Run tests
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True
        )
        
        output = result.stdout + result.stderr
        
        # Parse results
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        skipped = output.count(" SKIPPED")
        
        total_passed += passed
        total_failed += failed
        total_skipped += skipped
        
        results[suite_name] = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed
        }
        
        # Display results
        if passed + failed > 0:
            success_rate = (passed / (passed + failed)) * 100
            print(f"  ✅ Passed: {passed}")
            print(f"  ❌ Failed: {failed}")
            print(f"  ⏭️  Skipped: {skipped}")
            print(f"  📊 Success Rate: {success_rate:.1f}%")
        else:
            print(f"  ⚠️  No tests found or all skipped")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    
    total_tests = total_passed + total_failed
    if total_tests > 0:
        overall_success = (total_passed / total_tests) * 100
        
        print(f"\n✅ Total Passed: {total_passed}")
        print(f"❌ Total Failed: {total_failed}")
        print(f"⏭️  Total Skipped: {total_skipped}")
        print(f"\n🎯 Overall Success Rate: {overall_success:.1f}%")
        
        # Breakdown by suite
        print("\n📈 Suite Breakdown:")
        for suite, stats in results.items():
            if stats["total"] > 0:
                rate = (stats["passed"] / stats["total"]) * 100
                print(f"  {suite}: {rate:.1f}% ({stats['passed']}/{stats['total']})")
        
        # Status message
        print("\n" + "=" * 60)
        if overall_success == 100:
            print("🎉 PERFECT\! 100% TESTS PASSING\! 🎉")
            print("Your JARVIS system is fully tested and ready\!")
        elif overall_success >= 90:
            print("🚀 EXCELLENT\! Over 90% tests passing\!")
            print("You're ready to ship with high confidence\!")
        elif overall_success >= 80:
            print("💪 GREAT\! Over 80% tests passing\!")
            print("Core functionality is solid\!")
        elif overall_success >= 70:
            print("👍 GOOD\! Over 70% tests passing\!")
            print("Most features are working well\!")
        else:
            print("🔧 Keep improving\! Current pass rate: {:.1f}%".format(overall_success))
            print("Focus on the most critical failures first.")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.txt"
    
    with open(report_file, "w") as f:
        f.write(f"JARVIS Test Report - {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        
        for suite, stats in results.items():
            if stats["total"] > 0:
                rate = (stats["passed"] / stats["total"]) * 100
                f.write(f"{suite}:\n")
                f.write(f"  Success Rate: {rate:.1f}%\n")
                f.write(f"  Passed: {stats['passed']}\n")
                f.write(f"  Failed: {stats['failed']}\n")
                f.write(f"  Skipped: {stats['skipped']}\n\n")
        
        if total_tests > 0:
            f.write(f"\nOverall Success Rate: {overall_success:.1f}%\n")
    
    print(f"\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    run_test_suite()