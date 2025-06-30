#!/usr/bin/env python3
"""
JARVIS Implementation Progress Checker
Provides real-time metrics on implementation status
"""

import os
import glob
import ast
from pathlib import Path
from collections import defaultdict
import json


class ProgressChecker:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.stats = defaultdict(int)
        self.missing_implementations = []
        self.test_coverage = {}

    def check_progress(self):
        """Run all checks and generate report"""
        print("🔍 JARVIS Implementation Progress Check\n")

        self.check_core_components()
        self.check_tools_directory()
        self.check_utils_directory()
        self.check_test_coverage()
        self.check_missing_components()
        self.check_documentation()
        self.check_duplicates()

        self.generate_summary()

    def check_core_components(self):
        """Check core directory implementation status"""
        core_files = list(self.root_dir.glob("core/*.py"))
        self.stats["core_modules"] = len(
            [f for f in core_files if f.name != "__init__.py"]
        )

        # Check for actual implementations vs placeholders
        implemented = 0
        for file in core_files:
            if file.name == "__init__.py":
                continue
            content = file.read_text()
            # Simple heuristic: files with > 100 lines likely have real implementation
            if len(content.splitlines()) > 100:
                implemented += 1

        self.stats["core_implemented"] = implemented
        print(
            f"📦 Core Components: {implemented}/{self.stats['core_modules']} implemented"
        )

    def check_tools_directory(self):
        """Check tools directory population"""
        tools_files = list(self.root_dir.glob("tools/*.py"))
        tools_count = len([f for f in tools_files if f.name != "__init__.py"])
        self.stats["tools_count"] = tools_count

        print(f"🔧 Tools Directory: {tools_count} tools implemented")
        if tools_count == 0:
            print("   ⚠️  WARNING: Tools directory is empty!")

    def check_utils_directory(self):
        """Check utils directory population"""
        utils_files = list(self.root_dir.glob("utils/*.py"))
        utils_count = len([f for f in utils_files if f.name != "__init__.py"])
        self.stats["utils_count"] = utils_count

        print(f"🛠️  Utils Directory: {utils_count} utilities implemented")

    def check_test_coverage(self):
        """Estimate test coverage"""
        test_files = list(self.root_dir.glob("test_*.py"))
        self.stats["test_files"] = len(test_files)

        # Map tests to modules
        modules_with_tests = set()
        for test_file in test_files:
            # Extract module name from test file
            module_name = test_file.stem.replace("test_", "")
            modules_with_tests.add(module_name)

        # Calculate coverage estimate
        total_modules = self.stats["core_modules"]
        coverage_estimate = (len(modules_with_tests) / max(total_modules, 1)) * 100
        self.stats["test_coverage_estimate"] = coverage_estimate

        print(f"🧪 Test Coverage: {coverage_estimate:.1f}% (estimated)")
        print(f"   Test files: {self.stats['test_files']}")

    def check_missing_components(self):
        """Check missing_components.py for placeholders"""
        missing_file = self.root_dir / "missing_components.py"
        if missing_file.exists():
            content = missing_file.read_text()

            # Count placeholder classes
            try:
                tree = ast.parse(content)
                classes = [
                    node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                ]
                self.stats["placeholder_components"] = len(classes)

                # Check which have been implemented in core/
                implemented_in_core = 0
                for cls in classes:
                    core_file = self.root_dir / "core" / f"{cls.name.lower()}.py"
                    if core_file.exists():
                        implemented_in_core += 1

                self.stats["placeholders_implemented"] = implemented_in_core

                print(
                    f"🚧 Missing Components: {implemented_in_core}/{len(classes)} implemented"
                )

            except:
                print("   ⚠️  Could not parse missing_components.py")

    def check_documentation(self):
        """Check documentation status"""
        docs_files = list(self.root_dir.glob("docs/*.md"))
        api_docs = [f for f in docs_files if "api" in f.name.lower()]

        self.stats["docs_count"] = len(docs_files)
        self.stats["api_docs_count"] = len(api_docs)

        print(f"📚 Documentation: {len(docs_files)} files")
        print(f"   API docs: {len(api_docs)}")

    def check_duplicates(self):
        """Check for duplicate files"""
        all_py_files = list(self.root_dir.glob("**/*.py"))
        file_names = defaultdict(list)

        for file in all_py_files:
            if "__pycache__" not in str(file):
                file_names[file.name].append(str(file))

        duplicates = {
            name: paths for name, paths in file_names.items() if len(paths) > 1
        }
        self.stats["duplicate_files"] = len(duplicates)

        print(f"🔄 Duplicate Files: {len(duplicates)}")
        if duplicates:
            print("   Duplicates found:")
            for name, paths in list(duplicates.items())[:5]:  # Show first 5
                print(f"   - {name}: {len(paths)} copies")

    def generate_summary(self):
        """Generate overall summary"""
        print("\n📊 SUMMARY")
        print("=" * 50)

        # Calculate overall completion
        completion_factors = [
            (self.stats["core_implemented"] / max(self.stats["core_modules"], 1))
            * 0.4,  # 40% weight
            (min(self.stats["tools_count"], 10) / 10) * 0.2,  # 20% weight
            (self.stats["test_coverage_estimate"] / 100) * 0.2,  # 20% weight
            (min(self.stats["docs_count"], 10) / 10) * 0.1,  # 10% weight
            (
                self.stats["placeholders_implemented"]
                / max(self.stats["placeholder_components"], 1)
            )
            * 0.1,  # 10% weight
        ]

        overall_completion = sum(completion_factors) * 100

        print(f"Overall Completion: {overall_completion:.1f}%")
        print(f"Production Ready: {'Yes' if overall_completion > 80 else 'No'}")

        # Key metrics
        print(f"\nKey Metrics:")
        print(
            f"- Core Modules: {self.stats['core_implemented']}/{self.stats['core_modules']}"
        )
        print(f"- Tools: {self.stats['tools_count']}")
        print(f"- Utils: {self.stats['utils_count']}")
        print(f"- Test Coverage: {self.stats['test_coverage_estimate']:.1f}%")
        print(f"- Documentation Files: {self.stats['docs_count']}")
        print(f"- API Documentation: {self.stats['api_docs_count']}")
        print(f"- Duplicate Files: {self.stats['duplicate_files']}")

        # Recommendations
        print(f"\n🎯 Top Priorities:")
        priorities = []

        if self.stats["tools_count"] == 0:
            priorities.append(
                "1. Populate tools/ directory - Critical for functionality"
            )
        if self.stats["test_coverage_estimate"] < 50:
            priorities.append(
                "2. Increase test coverage - Currently too low for production"
            )
        if self.stats["api_docs_count"] == 0:
            priorities.append("3. Create API documentation - Essential for usage")
        if self.stats["duplicate_files"] > 5:
            priorities.append("4. Clean up duplicate files - Causing confusion")

        for priority in priorities[:5]:
            print(f"   {priority}")

        # Save report
        report = {
            "timestamp": str(Path.cwd()),
            "stats": dict(self.stats),
            "overall_completion": overall_completion,
            "production_ready": overall_completion > 80,
        }

        with open("progress_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to progress_report.json")


if __name__ == "__main__":
    checker = ProgressChecker()
    checker.check_progress()
