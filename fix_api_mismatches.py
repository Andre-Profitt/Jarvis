#!/usr/bin/env python3
"""
Fix API mismatches between tests and implementations
This aligns the actual code with the test expectations
"""

import os
import re
from pathlib import Path

def fix_database_api():
    """Fix database.py to match test expectations"""
    
    fixes = [
        # Add metadata parameter to add_message
        {
            "file": "core/database.py",
            "find": r"def add_message\(self, role: str, content: str, conversation_id: Optional[str] = None\)",
            "replace": "def add_message(self, role: str, content: str, conversation_id: Optional[str] = None, metadata: Optional[Dict] = None)"
        },
        
        # Add user_id to memory functions
        {
            "file": "core/database.py",
            "find": r"def store_memory\(self, content: str, memory_type: str = 'general'\)",
            "replace": "def store_memory(self, content: str, memory_type: str = 'general', user_id: Optional[str] = None)"
        },
        
        # Add user_id to record_learning
        {
            "file": "core/database.py",
            "find": r"def record_learning\(self, content: str, context: Optional[Dict] = None\)",
            "replace": "def record_learning(self, content: str, context: Optional[Dict] = None, user_id: Optional[str] = None)"
        }
    ]
    
    for fix in fixes:
        try:
            file_path = Path(fix["file"])
            if file_path.exists():
                content = file_path.read_text()
                updated = re.sub(fix["find"], fix["replace"], content)
                if updated != content:
                    file_path.write_text(updated)
                    print(f"✅ Fixed: {fix['file']}")
                else:
                    print(f"⚠️  Pattern not found in: {fix['file']}")
        except Exception as e:
            print(f"❌ Error fixing {fix['file']}: {e}")

def add_missing_methods():
    """Add methods that tests expect but don't exist"""
    
    # Add to configuration.py
    config_methods = '''
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path:
            self.config_file = config_path
        return self._load_config()
    
    def update(self, updates: Dict[str, Any], persist: bool = False):
        """Update configuration values"""
        self._update_nested(self.config, updates)
        if persist:
            self.save()
    
    def _update_nested(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested(d.get(k, {}), v)
            else:
                d[k] = v
        return d
'''
    
    print("\n📝 Add these methods to configuration.py if missing:")
    print(config_methods)

def main():
    print("🔧 Fixing API Mismatches\n")
    
    # Fix database API
    fix_database_api()
    
    # Show missing methods
    add_missing_methods()
    
    print("\n✅ Done! Next steps:")
    print("1. Review the changes")
    print("2. Run: pytest tests/test_database.py -v")
    print("3. Fix any remaining issues")

if __name__ == "__main__":
    main()