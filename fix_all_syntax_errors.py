#!/usr/bin/env python3
"""
Fix all syntax errors properly
"""

from pathlib import Path
import re

def fix_all_syntax_errors():
    """Fix all syntax errors in the project"""
    
    root = Path.cwd()
    fixed_files = []
    
    # Fix multi-ai-integration.py
    print("Fixing multi-ai-integration.py...")
    multi_ai_file = root / "multi-ai-integration.py"
    if multi_ai_file.exists():
        content = multi_ai_file.read_text()
        # Fix all except: blocks that are followed by logger.error using e
        content = re.sub(r'except:\s*\n\s*pass\s*\n\s*logger\.error\(f"([^"]+): \{e\}"\)', 
                        r'except Exception as e:\n            logger.error(f"\1: {e}")', content)
        # Fix standalone except: pass blocks
        content = re.sub(r'except:\s*\n\s*pass\s*\n', 
                        r'except Exception:\n            pass\n', content)
        # Fix malformed def _select_model line
        content = re.sub(r'def _select_model\(:', 
                        r'def _select_model(', content)
        multi_ai_file.write_text(content)
        fixed_files.append("multi-ai-integration.py")
    
    # Fix LAUNCH-JARVIS-REAL.py
    print("Fixing LAUNCH-JARVIS-REAL.py...")
    launch_file = root / "LAUNCH-JARVIS-REAL.py"
    if launch_file.exists():
        content = launch_file.read_text()
        # Fix except blocks
        content = re.sub(r'except:\s*\n\s*pass', 
                        r'except Exception:\n        pass', content)
        launch_file.write_text(content)
        fixed_files.append("LAUNCH-JARVIS-REAL.py")
    
    # Fix initial-training-data.py
    print("Fixing initial-training-data.py...")
    training_file = root / "initial-training-data.py"
    if training_file.exists():
        content = training_file.read_text()
        # Fix dictionary syntax errors - look for unclosed dictionaries
        lines = content.splitlines()
        fixed_lines = []
        in_dict = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            brace_count += line.count('{') - line.count('}')
            if brace_count > 0:
                in_dict = True
            elif brace_count == 0 and in_dict:
                in_dict = False
                # Make sure dictionary is properly closed
                if not line.strip().endswith(('}', '},', '],')):
                    line = line.rstrip() + '}'
            fixed_lines.append(line)
        
        training_file.write_text('\n'.join(fixed_lines))
        fixed_files.append("initial-training-data.py")
    
    # Fix core/enhanced_privacy_learning.py
    print("Fixing core/enhanced_privacy_learning.py...")
    privacy_file = root / "core" / "enhanced_privacy_learning.py"
    if privacy_file.exists():
        content = privacy_file.read_text()
        lines = content.splitlines()
        
        # Find line 751 and fix it
        if len(lines) > 750:
            # Check if it's an incomplete class or function
            if lines[750].strip() == "" or lines[750].strip().endswith(':'):
                lines[750] = "        pass  # TODO: Implement"
        
        privacy_file.write_text('\n'.join(lines))
        fixed_files.append("core/enhanced_privacy_learning.py")
    
    # Fix autonomous-tool-creation.py
    print("Fixing autonomous-tool-creation.py...")
    auto_file = root / "autonomous-tool-creation.py"
    if auto_file.exists():
        content = auto_file.read_text()
        lines = content.splitlines()
        
        # Fix line 384 and 409
        if len(lines) > 383:
            if lines[383].strip() == "" or lines[383].strip().endswith(':'):
                lines[383] = "            pass  # TODO: Implement"
        
        if len(lines) > 408:
            if 'logger.info(f"' in lines[408] and '{}' in lines[408]:
                lines[408] = '            logger.info("Tool creation completed")'
        
        auto_file.write_text('\n'.join(lines))
        fixed_files.append("autonomous-tool-creation.py")
    
    print(f"\n✅ Fixed {len(fixed_files)} files:")
    for f in fixed_files:
        print(f"  - {f}")
    
    # Now check for remaining errors
    print("\n🔍 Checking for remaining syntax errors...")
    error_count = 0
    
    for py_file in root.rglob("*.py"):
        try:
            compile(py_file.read_text(), py_file, 'exec')
        except SyntaxError as e:
            print(f"  ❌ {py_file.relative_to(root)}: Line {e.lineno} - {e.msg}")
            error_count += 1
    
    if error_count == 0:
        print("\n✅ All syntax errors fixed!")
    else:
        print(f"\n⚠️  {error_count} syntax errors remain")

if __name__ == "__main__":
    fix_all_syntax_errors()
