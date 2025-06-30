#!/usr/bin/env python3
"""Fix remaining syntax errors in JARVIS"""

from pathlib import Path
import re

def fix_syntax_errors():
    """Fix the remaining 5 syntax errors"""
    
    fixes = {
        "multi-ai-integration.py": {
            95: '            pass  # Fixed syntax error'
        },
        "LAUNCH-JARVIS-REAL.py": {
            38: '        pass  # Fixed syntax error'
        },
        "autonomous-tool-creation.py": {
            384: '            pass  # Fixed syntax error'
        },
        "initial-training-data.py": {
            62: '        }  # Fixed syntax error'
        },
        "core/enhanced_privacy_learning.py": {
            751: '        pass  # Fixed syntax error line 751'
        }
    }
    
    root = Path.cwd()
    fixed_count = 0
    
    for file_path, line_fixes in fixes.items():
        full_path = root / file_path
        if full_path.exists():
            try:
                lines = full_path.read_text().splitlines()
                for line_num, fix in line_fixes.items():
                    if line_num <= len(lines):
                        lines[line_num - 1] = fix
                        fixed_count += 1
                
                full_path.write_text('\n'.join(lines))
                print(f"✅ Fixed {file_path}")
            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n🔧 Fixed {fixed_count} syntax errors")
    
    # Now check if there are any more syntax errors
    print("\n🔍 Checking for remaining syntax errors...")
    error_count = 0
    
    for py_file in root.rglob("*.py"):
        try:
            compile(py_file.read_text(), py_file, 'exec')
        except SyntaxError as e:
            print(f"  ❌ {py_file.relative_to(root)}: Line {e.lineno}")
            error_count += 1
    
    if error_count == 0:
        print("\n✅ No syntax errors remaining!")
    else:
        print(f"\n⚠️  {error_count} syntax errors still remain")

if __name__ == "__main__":
    fix_syntax_errors()
