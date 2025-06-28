#!/usr/bin/env python3
"""
Dependency Health Check Script
Analyzes project dependencies for conflicts, vulnerabilities, and best practices
"""
import subprocess
import sys
import json
from pathlib import Path
import pkg_resources
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


class DependencyChecker:
    """Comprehensive dependency analysis tool"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues = defaultdict(list)
        self.stats = {
            "total_packages": 0,
            "direct_deps": 0,
            "transitive_deps": 0,
            "security_issues": 0,
            "version_conflicts": 0
        }
    
    def check_all(self):
        """Run all dependency checks"""
        print("🔍 JARVIS Dependency Health Check")
        print("=" * 50)
        
        self.check_installed_packages()
        self.check_version_conflicts()
        self.check_circular_dependencies()
        self.visualize_dependency_tree()
        self.check_security_vulnerabilities()
        self.check_unused_dependencies()
        self.generate_report()
    
    def check_installed_packages(self):
        """List all installed packages"""
        print("\n📦 Checking installed packages...")
        packages = list(pkg_resources.working_set)
        self.stats["total_packages"] = len(packages)
        
        # Categorize packages
        core_packages = []
        dev_packages = []
        
        for pkg in packages:
            if any(name in pkg.key for name in ['pytest', 'black', 'flake8', 'mypy', 'coverage']):
                dev_packages.append(pkg)
            else:
                core_packages.append(pkg)
        
        print(f"  ✓ Total packages: {len(packages)}")
        print(f"  ✓ Core packages: {len(core_packages)}")
        print(f"  ✓ Dev packages: {len(dev_packages)}")
    
    def check_version_conflicts(self):
        """Check for version conflicts"""
        print("\n🔧 Checking for version conflicts...")
        
        try:
            # Use pipdeptree to find conflicts
            result = subprocess.run(
                ['pipdeptree', '--warn', 'fail'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                conflicts = result.stderr.strip().split('\n')
                for conflict in conflicts:
                    if conflict and 'Warning' in conflict:
                        self.issues['conflicts'].append(conflict)
                        self.stats["version_conflicts"] += 1
                
                print(f"  ⚠️  Found {self.stats['version_conflicts']} conflicts")
            else:
                print("  ✓ No version conflicts detected")
        except FileNotFoundError:
            print("  ⚠️  pipdeptree not installed (pip install pipdeptree)")
    
    def check_circular_dependencies(self):
        """Check for circular dependencies"""
        print("\n🔄 Checking for circular dependencies...")
        
        try:
            result = subprocess.run(
                ['pipdeptree', '--warn', 'silence', '--json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                deps = json.loads(result.stdout)
                # Simple circular dependency detection
                circular = self._find_circular_deps(deps)
                if circular:
                    self.issues['circular'].extend(circular)
                    print(f"  ⚠️  Found {len(circular)} circular dependencies")
                else:
                    print("  ✓ No circular dependencies detected")
        except:
            print("  ⚠️  Could not check circular dependencies")
    
    def _find_circular_deps(self, deps):
        """Find circular dependencies in dependency tree"""
        circular = []
        visited = set()
        
        def visit(pkg_name, path):
            if pkg_name in path:
                circular.append(f"Circular: {' -> '.join(path)} -> {pkg_name}")
                return
            
            if pkg_name in visited:
                return
                
            visited.add(pkg_name)
            
            for dep in deps:
                if dep['package']['key'] == pkg_name:
                    for subdep in dep.get('dependencies', []):
                        visit(subdep['key'], path + [pkg_name])
        
        for dep in deps:
            visit(dep['package']['key'], [])
        
        return circular
    
    def visualize_dependency_tree(self):
        """Create a visual dependency tree"""
        print("\n🌳 Generating dependency tree...")
        
        try:
            # Generate tree for key packages
            key_packages = ['torch', 'transformers', 'fastapi', 'websockets']
            
            for pkg in key_packages:
                try:
                    result = subprocess.run(
                        ['pipdeptree', '-p', pkg],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print(f"\n  📦 {pkg} dependencies:")
                        for line in result.stdout.strip().split('\n')[:5]:
                            print(f"    {line}")
                except:
                    pass
        except:
            print("  ⚠️  Could not generate dependency tree")
    
    def check_security_vulnerabilities(self):
        """Check for known security vulnerabilities"""
        print("\n🔒 Checking security vulnerabilities...")
        
        try:
            # Try using safety
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    self.stats["security_issues"] = len(vulnerabilities)
                    for vuln in vulnerabilities:
                        self.issues['security'].append(
                            f"{vuln['package']} {vuln['installed_version']} - {vuln['description']}"
                        )
                    print(f"  ⚠️  Found {len(vulnerabilities)} security issues")
                else:
                    print("  ✓ No known vulnerabilities detected")
            else:
                print("  ✓ No security issues found")
        except FileNotFoundError:
            print("  ⚠️  safety not installed (pip install safety)")
        except:
            print("  ⚠️  Could not check security vulnerabilities")
    
    def check_unused_dependencies(self):
        """Find potentially unused dependencies"""
        print("\n🗑️  Checking for unused dependencies...")
        
        # This is a simplified check - in practice, you'd use tools like pip-autoremove
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            with open(requirements_file) as f:
                declared_deps = set(line.split('==')[0].split('>=')[0].strip() 
                                  for line in f if line.strip() and not line.startswith('#'))
            
            # Check which are imported in code
            imported = set()
            for py_file in self.project_root.rglob('*.py'):
                try:
                    with open(py_file) as f:
                        content = f.read()
                        for dep in declared_deps:
                            if f"import {dep}" in content or f"from {dep}" in content:
                                imported.add(dep)
                except:
                    pass
            
            unused = declared_deps - imported
            if unused:
                self.issues['unused'].extend(unused)
                print(f"  ⚠️  Found {len(unused)} potentially unused dependencies")
            else:
                print("  ✓ All dependencies appear to be used")
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 50)
        print("📊 DEPENDENCY HEALTH REPORT")
        print("=" * 50)
        
        print(f"\n📈 Statistics:")
        print(f"  • Total packages: {self.stats['total_packages']}")
        print(f"  • Version conflicts: {self.stats['version_conflicts']}")
        print(f"  • Security issues: {self.stats['security_issues']}")
        
        if any(self.issues.values()):
            print(f"\n⚠️  Issues Found:")
            
            if self.issues['conflicts']:
                print(f"\n  Version Conflicts:")
                for issue in self.issues['conflicts'][:5]:
                    print(f"    - {issue}")
            
            if self.issues['security']:
                print(f"\n  Security Vulnerabilities:")
                for issue in self.issues['security'][:5]:
                    print(f"    - {issue}")
            
            if self.issues['circular']:
                print(f"\n  Circular Dependencies:")
                for issue in self.issues['circular'][:5]:
                    print(f"    - {issue}")
            
            if self.issues['unused']:
                print(f"\n  Potentially Unused:")
                for pkg in list(self.issues['unused'])[:10]:
                    print(f"    - {pkg}")
        else:
            print("\n✅ No major issues found!")
        
        print("\n💡 Recommendations:")
        print("  1. Use 'poetry update' to update dependencies safely")
        print("  2. Run 'make security-scan' regularly")
        print("  3. Use 'poetry show --tree' to explore dependencies")
        print("  4. Consider using 'poetry export' for production")
        
        # Save detailed report
        report_path = self.project_root / 'dependency-report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'stats': self.stats,
                'issues': dict(self.issues),
                'timestamp': str(Path.ctime(Path.cwd()))
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    """Run dependency health check"""
    checker = DependencyChecker()
    checker.check_all()


if __name__ == "__main__":
    main()