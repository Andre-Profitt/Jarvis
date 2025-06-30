#!/bin/bash
# AI Test Generation System for JARVIS
# =====================================
# Systematically generates tests for components with low coverage

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🤖 AI Test Generation System${NC}"
echo "==========================="
echo ""

# Create necessary directories
mkdir -p tests/generated
mkdir -p tests/metadata

# Step 1: Analyze coverage gaps
analyze_coverage() {
    echo -e "${YELLOW}📊 Analyzing coverage gaps...${NC}"
    
    # Run coverage analysis quietly
    pytest --cov=core --cov=plugins --cov=tools --cov-report=json --quiet > /dev/null 2>&1 || true
    
    # Extract modules with low coverage
    python3 << 'EOF'
import json
import os

try:
    with open('coverage.json') as f:
        data = json.load(f)
    
    modules = []
    for file_path, stats in data['files'].items():
        # Skip test files and __pycache__
        if file_path.startswith('tests/') or '__pycache__' in file_path:
            continue
            
        coverage_pct = stats['summary']['percent_covered']
        if coverage_pct < 80:
            modules.append({
                'file': file_path,
                'coverage': coverage_pct,
                'missing_lines': len(stats.get('missing_lines', [])),
                'module_name': os.path.splitext(os.path.basename(file_path))[0]
            })
    
    # Sort by lowest coverage
    modules.sort(key=lambda x: x['coverage'])
    
    # Save to file for processing
    with open('tests/metadata/modules_to_test.json', 'w') as out:
        json.dump(modules, out, indent=2)
    
    print(f"Found {len(modules)} modules needing tests")
    print("\nTop 10 priorities:")
    print("-" * 50)
    for i, m in enumerate(modules[:10], 1):
        print(f"{i:2d}. {m['coverage']:5.1f}% - {m['file']}")
        print(f"    Missing {m['missing_lines']} lines of coverage")
    
except FileNotFoundError:
    print("No coverage.json found. Running initial coverage...")
    os.system("pytest --cov --cov-report=json --quiet")
EOF
}

# Step 2: Analyze module structure
analyze_module() {
    local module_path=$1
    local module_name=$(basename $module_path .py)
    
    echo -e "\n${YELLOW}🔍 Analyzing $module_name structure...${NC}"
    
    python3 << EOF
import ast
import json
import os

module_path = '$module_path'
module_name = '$module_name'

try:
    with open(module_path) as f:
        tree = ast.parse(f.read())
    
    # Extract structure
    structure = {
        'module_path': module_path,
        'module_name': module_name,
        'classes': [],
        'functions': [],
        'async_functions': [],
        'imports': [],
        'has_main': False
    }
    
    # Analyze AST
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'methods': [],
                'async_methods': [],
                'properties': []
            }
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name.startswith('_') and item.name != '__init__':
                        continue  # Skip private methods
                    class_info['methods'].append(item.name)
                elif isinstance(item, ast.AsyncFunctionDef):
                    if not item.name.startswith('_'):
                        class_info['async_methods'].append(item.name)
            
            structure['classes'].append(class_info)
            
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            if not node.name.startswith('_'):
                structure['functions'].append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef) and node.col_offset == 0:
            if not node.name.startswith('_'):
                structure['async_functions'].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                structure['imports'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            structure['imports'].append(node.module or '')
    
    # Check for main block
    for node in tree.body:
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Compare):
                if hasattr(node.test.left, 'id') and node.test.left.id == '__name__':
                    structure['has_main'] = True
    
    # Save structure
    with open(f'tests/metadata/{module_name}_structure.json', 'w') as f:
        json.dump(structure, f, indent=2)
    
    # Print summary
    print(f"Classes: {len(structure['classes'])}")
    for cls in structure['classes']:
        print(f"  - {cls['name']}: {len(cls['methods'])} methods, {len(cls['async_methods'])} async")
    print(f"Functions: {len(structure['functions'])}")
    print(f"Async Functions: {len(structure['async_functions'])}")
    
except Exception as e:
    print(f"Error analyzing module: {e}")
    
EOF
}

# Step 3: Generate test file from template
generate_test_file() {
    local module_path=$1
    local module_name=$(basename $module_path .py)
    local test_file="tests/test_${module_name}_generated.py"
    
    echo -e "\n${YELLOW}🔨 Generating test skeleton for $module_name...${NC}"
    
    # Copy template
    cp tests/test_generator_template.py "$test_file"
    
    # Read structure
    if [ -f "tests/metadata/${module_name}_structure.json" ]; then
        python3 << EOF
import json
import re

# Load structure
with open('tests/metadata/${module_name}_structure.json') as f:
    structure = json.load(f)

# Read template
with open('$test_file') as f:
    content = f.read()

# Replace placeholders
module_import_path = structure['module_path'].replace('.py', '').replace('/', '.')
content = content.replace('{MODULE_NAME}', structure['module_name'])
content = content.replace('{module_path}', module_import_path)

# Handle main class
if structure['classes']:
    main_class = structure['classes'][0]['name']
    content = content.replace('{ComponentClass}', main_class)
    
    # Add specific method tests
    test_methods = []
    
    for method in structure['classes'][0]['methods']:
        if method != '__init__':
            test_methods.append(f'''
    def test_{method}(self, component):
        """Test {method} functionality"""
        # TODO: Implement test for {method}
        result = component.{method}()
        assert result is not None
''')
    
    for async_method in structure['classes'][0]['async_methods']:
        test_methods.append(f'''
    @pytest.mark.asyncio
    async def test_{async_method}(self, component):
        """Test async {async_method}"""
        # TODO: Implement test for {async_method}
        result = await component.{async_method}()
        assert result is not None
''')
    
    # Insert additional tests before the template markers
    content = content.replace('{method_name}', structure['classes'][0]['methods'][0] if structure['classes'][0]['methods'] else 'process')
    content = content.replace('{async_method}', structure['classes'][0]['async_methods'][0] if structure['classes'][0]['async_methods'] else 'async_process')
else:
    # Module with only functions
    content = content.replace('{ComponentClass}', 'None')
    content = content.replace('component.', '')

# Save generated file
with open('$test_file', 'w') as f:
    f.write(content)

print(f"Generated: $test_file")
print(f"  - Based on {len(structure['classes'])} classes")
print(f"  - Includes tests for {sum(len(c['methods']) + len(c['async_methods']) for c in structure['classes'])} methods")

EOF
    else
        echo -e "${RED}❌ No structure metadata found for $module_name${NC}"
    fi
}

# Step 4: Validate generated tests
validate_tests() {
    local test_file=$1
    local module_name=$(basename $test_file .py | sed 's/test_//' | sed 's/_generated//')
    
    echo -e "\n${YELLOW}🧪 Validating $test_file...${NC}"
    
    # Check syntax
    if python3 -m py_compile "$test_file" 2>/dev/null; then
        echo -e "${GREEN}✅ Syntax valid${NC}"
    else
        echo -e "${RED}❌ Syntax error in generated test${NC}"
        return 1
    fi
    
    # Check if tests can be collected
    test_count=$(pytest "$test_file" --collect-only -q 2>/dev/null | grep -c "<Function" || true)
    echo -e "${GREEN}✅ Found $test_count test functions${NC}"
    
    # Run tests (allow failures for TODO tests)
    echo -e "\n${BLUE}Running generated tests...${NC}"
    pytest "$test_file" -v --tb=short || true
    
    # Check coverage improvement
    echo -e "\n${BLUE}Checking coverage impact...${NC}"
    if [ -f "core/${module_name}.py" ]; then
        pytest "$test_file" --cov="core.${module_name}" --cov-report=term || true
    fi
}

# Step 5: Generate batch summary
generate_summary() {
    echo -e "\n${BLUE}📊 Generating test generation summary...${NC}"
    
    python3 << 'EOF'
import json
import glob
import os

# Count generated tests
generated_tests = glob.glob('tests/test_*_generated.py')
print(f"\n✅ Generated {len(generated_tests)} test files")

# Load coverage gaps
if os.path.exists('tests/metadata/modules_to_test.json'):
    with open('tests/metadata/modules_to_test.json') as f:
        modules = json.load(f)
    
    print(f"\n📈 Coverage improvement potential:")
    print("-" * 50)
    
    total_missing = sum(m['missing_lines'] for m in modules[:10])
    print(f"Total missing lines in top 10 modules: {total_missing}")
    
    # Calculate potential coverage increase
    current_avg = sum(m['coverage'] for m in modules[:10]) / 10
    print(f"Current average coverage: {current_avg:.1f}%")
    print(f"Potential coverage gain: {80 - current_avg:.1f}%")

print("\n🎯 Next steps:")
print("1. Review generated test files in tests/*_generated.py")
print("2. Replace TODO comments with actual test logic")
print("3. Run tests and verify coverage improvement")
print("4. Move completed tests to final names (remove '_generated')")
EOF
}

# Main execution flow
main() {
    case "${1:-all}" in
        analyze)
            analyze_coverage
            ;;
        generate)
            if [ -z "$2" ]; then
                echo "Usage: $0 generate <module_path>"
                exit 1
            fi
            analyze_module "$2"
            generate_test_file "$2"
            validate_tests "tests/test_$(basename $2 .py)_generated.py"
            ;;
        batch)
            # Batch process top N modules
            analyze_coverage
            
            # Process top 5 modules
            echo -e "\n${BLUE}🚀 Batch processing top 5 modules...${NC}"
            python3 -c "
import json
with open('tests/metadata/modules_to_test.json') as f:
    modules = json.load(f)
    for m in modules[:5]:
        print(m['file'])
" | while read -r module; do
                echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                analyze_module "$module"
                generate_test_file "$module"
                validate_tests "tests/test_$(basename $module .py)_generated.py"
            done
            
            generate_summary
            ;;
        validate)
            if [ -z "$2" ]; then
                # Validate all generated tests
                for test_file in tests/test_*_generated.py; do
                    validate_tests "$test_file"
                done
            else
                validate_tests "$2"
            fi
            ;;
        summary)
            generate_summary
            ;;
        clean)
            echo "🧹 Cleaning generated files..."
            rm -f tests/test_*_generated.py
            rm -rf tests/metadata
            rm -f coverage.json
            echo "✅ Cleaned"
            ;;
        *)
            echo "AI Test Generator - Usage:"
            echo "  $0 analyze              - Analyze coverage gaps"
            echo "  $0 generate <module>    - Generate tests for specific module"
            echo "  $0 batch                - Batch generate for top 5 modules"
            echo "  $0 validate [test_file] - Validate generated tests"
            echo "  $0 summary              - Show generation summary"
            echo "  $0 clean                - Clean generated files"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"