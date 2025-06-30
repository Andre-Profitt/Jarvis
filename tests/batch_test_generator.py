#!/usr/bin/env python3
"""
Batch Test Generator for JARVIS
===============================
Processes multiple modules systematically to generate comprehensive test suites.
Integrates with existing test infrastructure and tracks progress.
"""

import json
import subprocess
import sys
from pathlib import Path
import ast
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import re


class TestGeneratorBatch:
    """Batch processor for test generation"""
    
    def __init__(self, target_coverage: float = 80.0):
        self.target_coverage = target_coverage
        self.template_path = Path("tests/test_generator_template.py")
        self.metadata_dir = Path("tests/metadata")
        self.generated_dir = Path("tests/generated")
        self.results = []
        
        # Create directories
        self.metadata_dir.mkdir(exist_ok=True)
        self.generated_dir.mkdir(exist_ok=True)
        
        # Load template
        if self.template_path.exists():
            self.template = self.template_path.read_text()
        else:
            raise FileNotFoundError(f"Template not found: {self.template_path}")
    
    def get_low_coverage_modules(self) -> List[Dict]:
        """Get modules that need tests"""
        print("📊 Analyzing current coverage...")
        
        # Run coverage analysis
        result = subprocess.run(
            ["pytest", "--cov=core", "--cov=plugins", "--cov=tools", 
             "--cov-report=json", "--quiet"],
            capture_output=True,
            text=True
        )
        
        if not Path("coverage.json").exists():
            print("⚠️  No coverage data found. Running initial coverage...")
            subprocess.run(["pytest", "--cov", "--cov-report=json", "--quiet"])
        
        with open("coverage.json") as f:
            data = json.load(f)
        
        modules = []
        for file_path, stats in data['files'].items():
            # Skip test files and other non-source files
            if (file_path.startswith('tests/') or 
                '__pycache__' in file_path or
                file_path.endswith('__init__.py')):
                continue
            
            coverage_pct = stats['summary']['percent_covered']
            if coverage_pct < self.target_coverage:
                modules.append({
                    'path': file_path,
                    'coverage': coverage_pct,
                    'missing_lines': len(stats.get('missing_lines', [])),
                    'missing_branches': len(stats.get('missing_branches', [])),
                    'module_name': Path(file_path).stem
                })
        
        # Sort by lowest coverage
        modules.sort(key=lambda x: x['coverage'])
        
        # Save analysis
        analysis_file = self.metadata_dir / "coverage_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'target_coverage': self.target_coverage,
                'modules_below_target': len(modules),
                'modules': modules
            }, f, indent=2)
        
        return modules
    
    def analyze_module(self, module_path: str) -> Dict:
        """Extract detailed module structure"""
        print(f"  🔍 Analyzing {module_path}...")
        
        try:
            with open(module_path) as f:
                source_code = f.read()
                tree = ast.parse(source_code)
        except Exception as e:
            print(f"  ❌ Error parsing {module_path}: {e}")
            return None
        
        structure = {
            'module_path': module_path,
            'module_name': Path(module_path).stem,
            'classes': [],
            'functions': [],
            'async_functions': [],
            'imports': set(),
            'external_dependencies': set(),
            'has_main': False,
            'decorators_used': set(),
            'complexity_estimate': 0
        }
        
        class StructureVisitor(ast.NodeVisitor):
            def __init__(self, structure):
                self.structure = structure
                self.current_class = None
                
            def visit_ClassDef(self, node):
                class_info = {
                    'name': node.name,
                    'methods': [],
                    'async_methods': [],
                    'properties': [],
                    'class_methods': [],
                    'static_methods': [],
                    'base_classes': [self.get_name(base) for base in node.bases],
                    'decorators': [self.get_name(d) for d in node.decorator_list]
                }
                
                self.current_class = class_info
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'decorators': [self.get_name(d) for d in item.decorator_list]
                        }
                        
                        # Categorize methods
                        if item.name.startswith('_') and item.name != '__init__':
                            continue  # Skip private methods
                        
                        if '@property' in str(item.decorator_list):
                            class_info['properties'].append(item.name)
                        elif '@classmethod' in str(item.decorator_list):
                            class_info['class_methods'].append(method_info)
                        elif '@staticmethod' in str(item.decorator_list):
                            class_info['static_methods'].append(method_info)
                        else:
                            class_info['methods'].append(method_info)
                    
                    elif isinstance(item, ast.AsyncFunctionDef):
                        if not item.name.startswith('_'):
                            class_info['async_methods'].append({
                                'name': item.name,
                                'args': [arg.arg for arg in item.args.args]
                            })
                
                self.structure['classes'].append(class_info)
                self.structure['complexity_estimate'] += len(class_info['methods']) + len(class_info['async_methods'])
                self.current_class = None
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if self.current_class is None and node.col_offset == 0:
                    if not node.name.startswith('_'):
                        self.structure['functions'].append({
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'decorators': [self.get_name(d) for d in node.decorator_list]
                        })
                        self.structure['complexity_estimate'] += 1
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                if self.current_class is None and node.col_offset == 0:
                    if not node.name.startswith('_'):
                        self.structure['async_functions'].append({
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args]
                        })
                        self.structure['complexity_estimate'] += 1
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    self.structure['imports'].add(alias.name)
                    # Track external dependencies
                    if not alias.name.startswith('.'):
                        self.structure['external_dependencies'].add(alias.name.split('.')[0])
            
            def visit_ImportFrom(self, node):
                if node.module:
                    self.structure['imports'].add(node.module)
                    if not node.module.startswith('.'):
                        self.structure['external_dependencies'].add(node.module.split('.')[0])
            
            def visit_If(self, node):
                # Check for main block
                if (isinstance(node.test, ast.Compare) and 
                    isinstance(node.test.left, ast.Name) and 
                    node.test.left.id == '__name__'):
                    self.structure['has_main'] = True
                self.generic_visit(node)
            
            def get_name(self, node):
                """Extract name from AST node"""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self.get_name(node.value)}.{node.attr}"
                elif isinstance(node, ast.Call):
                    return self.get_name(node.func)
                return str(node)
        
        visitor = StructureVisitor(structure)
        visitor.visit(tree)
        
        # Convert sets to lists for JSON serialization
        structure['imports'] = list(structure['imports'])
        structure['external_dependencies'] = list(structure['external_dependencies'])
        structure['decorators_used'] = list(structure['decorators_used'])
        
        # Save structure metadata
        metadata_file = self.metadata_dir / f"{structure['module_name']}_structure.json"
        with open(metadata_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        return structure
    
    def generate_test_outline(self, module_path: str, structure: Dict) -> Optional[str]:
        """Generate comprehensive test file outline"""
        if not structure:
            return None
        
        module_name = structure['module_name']
        print(f"  🔨 Generating tests for {module_name}...")
        
        # Start with template
        test_content = self.template
        
        # Replace basic placeholders
        module_import_path = module_path.replace('.py', '').replace('/', '.')
        test_content = test_content.replace('{MODULE_NAME}', module_name)
        test_content = test_content.replace('{module_path}', module_import_path)
        
        # Handle classes
        if structure['classes']:
            main_class = structure['classes'][0]
            test_content = test_content.replace('{ComponentClass}', main_class['name'])
            
            # Generate specific test methods
            additional_tests = []
            
            # Add tests for each class
            for cls in structure['classes']:
                if cls != main_class:
                    additional_tests.append(f"\n\nclass Test{cls['name']}:")
                    additional_tests.append(f'    """Test suite for {cls["name"]}"""')
                
                # Method tests
                for method in cls['methods']:
                    if method['name'] == '__init__':
                        continue
                    
                    test_name = f"test_{method['name']}"
                    args_str = ', '.join(method['args'][1:])  # Skip 'self'
                    
                    additional_tests.append(f'''
    def {test_name}(self, component):
        """Test {method['name']} with various inputs"""
        # TODO: Test with valid inputs
        result = component.{method['name']}({args_str})
        assert result is not None
        
        # TODO: Test edge cases
        # TODO: Test error conditions''')
                
                # Async method tests
                for method in cls['async_methods']:
                    test_name = f"test_{method['name']}"
                    args_str = ', '.join(method['args'][1:])  # Skip 'self'
                    
                    additional_tests.append(f'''
    @pytest.mark.asyncio
    async def {test_name}(self, component):
        """Test async {method['name']}"""
        # TODO: Test with valid inputs
        result = await component.{method['name']}({args_str})
        assert result is not None
        
        # TODO: Test concurrent execution
        # TODO: Test timeout scenarios''')
                
                # Property tests
                for prop in cls['properties']:
                    additional_tests.append(f'''
    def test_{prop}_property(self, component):
        """Test {prop} property"""
        # TODO: Test getter
        value = component.{prop}
        assert value is not None
        
        # TODO: Test setter if applicable''')
            
            # Add integration test for complex modules
            if structure['complexity_estimate'] > 10:
                additional_tests.append('''
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow(self, component):
        """Test complete workflow integration"""
        # TODO: Implement end-to-end test
        pass''')
            
            # Insert additional tests
            test_content += '\n' + '\n'.join(additional_tests)
        
        else:
            # Module with only functions
            test_content = test_content.replace('{ComponentClass}', 'None')
            test_content = test_content.replace('from {module_path} import {ComponentClass}', 
                                              f'import {module_import_path}')
            
            # Generate function tests
            function_tests = ['\n# ===== Function Tests =====']
            
            for func in structure['functions']:
                args_str = ', '.join(func['args'])
                function_tests.append(f'''
def test_{func["name"]}():
    """Test {func["name"]} function"""
    # TODO: Import function
    from {module_import_path} import {func["name"]}
    
    # TODO: Test with valid inputs
    result = {func["name"]}({args_str})
    assert result is not None
    
    # TODO: Test edge cases''')
            
            for func in structure['async_functions']:
                args_str = ', '.join(func['args'])
                function_tests.append(f'''
@pytest.mark.asyncio
async def test_{func["name"]}():
    """Test async {func["name"]} function"""
    from {module_import_path} import {func["name"]}
    
    result = await {func["name"]}({args_str})
    assert result is not None''')
            
            test_content += '\n' + '\n'.join(function_tests)
        
        # Add dependency mocking hints
        if structure['external_dependencies']:
            mock_hints = ['\n# ===== Mock External Dependencies =====']
            mock_hints.append('# TODO: Mock these external dependencies:')
            for dep in sorted(structure['external_dependencies']):
                if dep in ['numpy', 'pandas', 'requests', 'aiohttp', 'asyncio']:
                    mock_hints.append(f'# - {dep}: Use fixtures from conftest.py')
                else:
                    mock_hints.append(f'# - {dep}: Create mock in test file')
            
            test_content += '\n' + '\n'.join(mock_hints)
        
        # Save generated test
        test_path = self.generated_dir / f"test_{module_name}_generated.py"
        with open(test_path, 'w') as f:
            f.write(test_content)
        
        return str(test_path)
    
    def validate_generated_test(self, test_path: str) -> Dict:
        """Validate and run generated test"""
        print(f"  🧪 Validating {Path(test_path).name}...")
        
        validation_result = {
            'test_file': test_path,
            'syntax_valid': False,
            'imports_valid': False,
            'tests_collected': 0,
            'tests_passed': 0,
            'coverage_impact': 0.0
        }
        
        # Check syntax
        result = subprocess.run(
            ["python3", "-m", "py_compile", test_path],
            capture_output=True,
            text=True
        )
        validation_result['syntax_valid'] = result.returncode == 0
        
        if not validation_result['syntax_valid']:
            print(f"    ❌ Syntax error: {result.stderr}")
            return validation_result
        
        # Collect tests
        result = subprocess.run(
            ["pytest", test_path, "--collect-only", "-q"],
            capture_output=True,
            text=True
        )
        
        test_count = len([line for line in result.stdout.split('\n') if '<Function' in line])
        validation_result['tests_collected'] = test_count
        print(f"    ✅ Collected {test_count} tests")
        
        # Try to run tests (expect some failures due to TODOs)
        result = subprocess.run(
            ["pytest", test_path, "-v", "--tb=no", "-q"],
            capture_output=True,
            text=True
        )
        
        # Count passed tests
        passed = result.stdout.count(' PASSED')
        validation_result['tests_passed'] = passed
        
        return validation_result
    
    def process_module(self, module_info: Dict) -> Dict:
        """Process a single module"""
        result = {
            'module': module_info['path'],
            'original_coverage': module_info['coverage'],
            'structure': None,
            'test_file': None,
            'validation': None,
            'success': False
        }
        
        try:
            # Analyze structure
            structure = self.analyze_module(module_info['path'])
            result['structure'] = structure
            
            if structure:
                # Generate test
                test_file = self.generate_test_outline(module_info['path'], structure)
                result['test_file'] = test_file
                
                if test_file:
                    # Validate
                    validation = self.validate_generated_test(test_file)
                    result['validation'] = validation
                    result['success'] = validation['syntax_valid']
        
        except Exception as e:
            print(f"    ❌ Error processing {module_info['path']}: {e}")
            result['error'] = str(e)
        
        return result
    
    def process_batch(self, max_modules: int = 10, parallel: bool = False) -> List[Dict]:
        """Process multiple modules"""
        print("\n🚀 Starting batch test generation...")
        
        # Get modules needing tests
        modules = self.get_low_coverage_modules()
        
        if not modules:
            print("✅ All modules have sufficient coverage!")
            return []
        
        # Limit to max_modules
        modules_to_process = modules[:max_modules]
        
        print(f"\n📋 Processing {len(modules_to_process)} modules:")
        for i, m in enumerate(modules_to_process, 1):
            print(f"  {i:2d}. {m['coverage']:5.1f}% - {m['path']}")
        
        print("\n" + "="*60)
        
        # Process modules
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.process_module, m) for m in modules_to_process]
                self.results = [f.result() for f in concurrent.futures.as_completed(futures)]
        else:
            self.results = []
            for i, module in enumerate(modules_to_process, 1):
                print(f"\n[{i}/{len(modules_to_process)}] Processing {module['module_name']}...")
                result = self.process_module(module)
                self.results.append(result)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save processing results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'modules_processed': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'total_tests_generated': sum(r['validation']['tests_collected'] 
                                       for r in self.results 
                                       if r.get('validation')),
            'results': self.results
        }
        
        summary_file = self.metadata_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 Test Generation Summary:")
        print(f"  Modules processed: {summary['modules_processed']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Tests generated: {summary['total_tests_generated']}")
        print(f"\n📁 Generated tests in: {self.generated_dir}")
        print(f"📋 Summary saved to: {summary_file}")
    
    def finalize_tests(self):
        """Move generated tests to final location"""
        print("\n🏁 Finalizing tests...")
        
        for test_file in self.generated_dir.glob("test_*_generated.py"):
            final_name = test_file.name.replace('_generated', '')
            final_path = Path('tests') / final_name
            
            if final_path.exists():
                print(f"  ⚠️  {final_name} already exists, keeping as {test_file.name}")
            else:
                shutil.move(str(test_file), str(final_path))
                print(f"  ✅ Moved to {final_path}")
    
    def coverage_report(self):
        """Generate coverage improvement report"""
        print("\n📈 Running coverage analysis...")
        
        # Run coverage with all tests
        subprocess.run([
            "pytest", "--cov=core", "--cov=plugins", "--cov=tools",
            "--cov-report=term", "--cov-report=html"
        ])


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Test Generator for JARVIS')
    parser.add_argument('--max', type=int, default=10, 
                       help='Maximum number of modules to process')
    parser.add_argument('--parallel', action='store_true', 
                       help='Process modules in parallel')
    parser.add_argument('--target', type=float, default=80.0,
                       help='Target coverage percentage')
    parser.add_argument('--finalize', action='store_true',
                       help='Move generated tests to final location')
    parser.add_argument('--report', action='store_true',
                       help='Generate coverage report after processing')
    
    args = parser.parse_args()
    
    generator = TestGeneratorBatch(target_coverage=args.target)
    
    # Process batch
    results = generator.process_batch(max_modules=args.max, parallel=args.parallel)
    
    # Finalize if requested
    if args.finalize and results:
        generator.finalize_tests()
    
    # Generate report if requested
    if args.report:
        generator.coverage_report()


if __name__ == "__main__":
    main()