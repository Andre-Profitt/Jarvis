#!/usr/bin/env python3
"""
Fix all syntax errors in JARVIS ecosystem
This script will attempt to fix common syntax errors across all Python files
"""
import os
import ast
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SyntaxFixer:
    def __init__(self, project_root="/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM"):
        self.project_root = Path(project_root)
        self.errors_fixed = 0
        self.files_processed = 0
        
    def fix_file(self, filepath):
        """Attempt to fix syntax errors in a Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # Common fixes
            fixes = [
                # Fix f-string syntax errors
                (r'f"([^"]*){([^}]*)}([^"]*)"', self._fix_fstring),
                
                # Fix missing colons
                (r'^(\s*)(if|elif|else|for|while|def|class|try|except|finally|with)([^:]+)$', 
                 r'\1\2\3:', re.MULTILINE),
                
                # Fix unclosed brackets
                (r'(\[)[^\]]*$', self._fix_unclosed_bracket),
                
                # Fix incomplete imports
                (r'^from\s+\w+\s*$', r'# Incomplete import removed', re.MULTILINE),
                (r'^import\s*$', r'# Incomplete import removed', re.MULTILINE),
                
                # Fix trailing commas in function definitions
                (r'def\s+\w+\([^)]*,\s*\)', self._fix_trailing_comma),
                
                # Fix empty except blocks
                (r'except[^:]*:\s*$', r'except:\n    pass', re.MULTILINE),
            ]
            
            for pattern, replacement, *flags in fixes:
                if callable(replacement):
                    content = replacement(content, pattern)
                else:
                    flag = flags[0] if flags else 0
                    content = re.sub(pattern, replacement, content, flags=flag)
            
            # Try to parse the fixed content
            try:
                ast.parse(content)
                
                # If parsing succeeds and changes were made, save the file
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.errors_fixed += 1
                    logger.info(f"✅ Fixed: {filepath}")
                    return True
                    
            except SyntaxError as e:
                # If still has errors, log them
                logger.debug(f"❌ Still has errors: {filepath} - {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return False
    
    def _fix_fstring(self, content, pattern):
        """Fix f-string formatting issues"""
        def replacer(match):
            prefix, expr, suffix = match.groups()
            # Ensure expressions in f-strings are properly formatted
            expr = expr.strip()
            return f'f"{prefix}{{{expr}}}{suffix}"'
        
        return re.sub(pattern, replacer, content)
    
    def _fix_unclosed_bracket(self, content, pattern):
        """Fix unclosed brackets"""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if re.search(r'(\[)[^\]]*$', line):
                line += ']'
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def _fix_trailing_comma(self, content, pattern):
        """Fix trailing commas in function definitions"""
        return re.sub(r'(def\s+\w+\([^)]*),\s*\)', r'\1)', content)
    
    def fix_all_files(self):
        """Fix all Python files in the project"""
        logger.info("🔧 Starting syntax error fixes...")
        
        python_files = list(self.project_root.rglob("*.py"))
        total_files = len(python_files)
        
        for i, filepath in enumerate(python_files, 1):
            # Skip virtual environments and cache
            if any(part in str(filepath) for part in ['.venv', '__pycache__', 'node_modules']):
                continue
                
            logger.info(f"Processing ({i}/{total_files}): {filepath.name}")
            self.files_processed += 1
            self.fix_file(filepath)
        
        logger.info(f"\n✨ Fixing complete!")
        logger.info(f"📊 Files processed: {self.files_processed}")
        logger.info(f"✅ Errors fixed: {self.errors_fixed}")

def main():
    """Main function"""
    fixer = SyntaxFixer()
    fixer.fix_all_files()
    
    # After fixing, try to launch the full JARVIS
    logger.info("\n🚀 Attempting to launch full JARVIS...")
    os.system("cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM && python3 LAUNCH-JARVIS-REAL.py")

if __name__ == "__main__":
    main()
