#!/usr/bin/env python3
"""
Find and report on TODO items in test files
Helps identify where test implementations need to be completed
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def find_todos_in_tests():
    """Find all TODO comments in test files"""
    test_dir = Path("tests")
    todos = defaultdict(list)
    
    # Pattern to match TODO comments
    todo_pattern = re.compile(r'#\s*TODO:?\s*(.*)|""".*TODO:?\s*(.*?)"""', re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    for test_file in test_dir.glob("test_*.py"):
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Find all TODOs
        matches = todo_pattern.finditer(content)
        for match in matches:
            # Get line number
            line_num = content[:match.start()].count('\n') + 1
            todo_text = match.group(1) or match.group(2) or "No description"
            todos[test_file.name].append({
                'line': line_num,
                'text': todo_text.strip()
            })
    
    return todos

def generate_todo_report(todos):
    """Generate a report of all TODOs"""
    total_todos = sum(len(items) for items in todos.values())
    
    print("🔍 TODO Report for Test Files")
    print("=" * 60)
    print(f"\n📊 Summary: {total_todos} TODOs found in {len(todos)} files\n")
    
    if total_todos == 0:
        print("✅ No TODOs found! All tests are complete.")
        return
    
    # Group by priority (inferred from content)
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for filename, todo_items in sorted(todos.items()):
        print(f"\n📄 {filename}:")
        for todo in todo_items:
            print(f"   Line {todo['line']}: {todo['text']}")
            
            # Categorize by keywords
            text_lower = todo['text'].lower()
            if any(word in text_lower for word in ['implement', 'fix', 'critical', 'must']):
                high_priority.append((filename, todo))
            elif any(word in text_lower for word in ['add', 'create', 'should']):
                medium_priority.append((filename, todo))
            else:
                low_priority.append((filename, todo))
    
    # Priority summary
    print("\n\n🎯 Priority Breakdown:")
    print(f"   🔴 High Priority: {len(high_priority)} TODOs")
    print(f"   🟡 Medium Priority: {len(medium_priority)} TODOs")
    print(f"   🟢 Low Priority: {len(low_priority)} TODOs")
    
    # Quick fixes
    print("\n\n💡 Quick Fix Suggestions:")
    print("1. Mock external dependencies instead of implementing them")
    print("2. Use pytest fixtures for common test data")
    print("3. Focus on testing core functionality first")
    print("4. Add integration tests after unit tests are complete")
    
    # Generate fix script
    generate_fix_script(todos)

def generate_fix_script(todos):
    """Generate a script to help fix TODOs"""
    script_content = '''#!/usr/bin/env python3
"""
Auto-generated script to help implement remaining test TODOs
"""

import os

# Common test implementations
TEST_TEMPLATES = {
    "mock_setup": """
        # Mock external dependencies
        mock_client = Mock()
        mock_client.method.return_value = "expected_result"
    """,
    
    "async_test": """
        @pytest.mark.asyncio
        async def test_async_function():
            result = await async_function()
            assert result is not None
    """,
    
    "error_test": """
        def test_error_handling():
            with pytest.raises(ValueError):
                function_that_should_fail()
    """
}

def implement_todos():
    """Implement common TODO patterns"""
    # This is a template - customize for your needs
    pass

if __name__ == "__main__":
    implement_todos()
'''
    
    with open("implement_test_todos.py", "w") as f:
        f.write(script_content)
    
    print(f"\n✅ Generated 'implement_test_todos.py' to help with implementations")

def main():
    """Main function"""
    todos = find_todos_in_tests()
    generate_todo_report(todos)
    
    # Also check for empty test functions
    print("\n\n🔍 Checking for empty test functions...")
    empty_tests = find_empty_tests()
    if empty_tests:
        print(f"⚠️  Found {len(empty_tests)} empty test functions:")
        for file, func in empty_tests[:5]:  # Show first 5
            print(f"   {file}: {func}")
        if len(empty_tests) > 5:
            print(f"   ... and {len(empty_tests) - 5} more")
    else:
        print("✅ No empty test functions found!")

def find_empty_tests():
    """Find test functions that only contain 'pass' or are empty"""
    test_dir = Path("tests")
    empty_tests = []
    
    # Pattern to match test functions
    test_func_pattern = re.compile(r'def (test_\w+)\([^)]*\):\s*(?:"""[^"]*"""\s*)?(?:pass|\.\.\.|\s*$)', re.MULTILINE)
    
    for test_file in test_dir.glob("test_*.py"):
        with open(test_file, 'r') as f:
            content = f.read()
        
        matches = test_func_pattern.findall(content)
        for match in matches:
            empty_tests.append((test_file.name, match))
    
    return empty_tests

if __name__ == "__main__":
    main()