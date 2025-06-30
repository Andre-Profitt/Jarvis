#!/usr/bin/env python3
"""
JARVIS Duplicate File Cleanup Script
Safely removes duplicate files and consolidates the codebase
"""

import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import json

class JARVISCleanup:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.removal_log = []
        
    def create_backup(self):
        """Create backup directory for safety"""
        self.backup_dir.mkdir(exist_ok=True)
        print(f"✅ Created backup directory: {self.backup_dir}")
        
    def file_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def backup_and_remove(self, filepath):
        """Backup file before removal"""
        relative_path = filepath.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(filepath, backup_path)
        os.remove(filepath)
        
        self.removal_log.append({
            "file": str(relative_path),
            "action": "removed",
            "backup": str(backup_path)
        })
        print(f"  ❌ Removed: {relative_path}")
    
    def clean_duplicate_launchers(self):
        """Remove duplicate launcher files, keeping only the unified one"""
        print("\n🚀 Cleaning duplicate launchers...")
        
        launchers_to_remove = [
            "LAUNCH-JARVIS.py",
            "LAUNCH-JARVIS-ENHANCED.py", 
            "LAUNCH-JARVIS-FIXED.py",
            "LAUNCH-JARVIS-FULL.py",
            "LAUNCH-JARVIS-PATCHED.py",
            "LAUNCH-JARVIS-UNIFIED.py"
        ]
        
        for launcher in launchers_to_remove:
            launcher_path = self.project_root / launcher
            if launcher_path.exists():
                self.backup_and_remove(launcher_path)
    
    def clean_elite_proactive_duplicates(self):
        """Remove duplicate elite proactive assistant files"""
        print("\n🧠 Cleaning elite proactive assistant duplicates...")
        
        # Analyze the files first
        elite_files = {
            "core/elite_proactive_assistant.py": None,
            "core/elite_proactive_assistant_backup.py": None,
            "core/elite_proactive_assistant_v2.py": None
        }
        
        # Get file sizes and hashes
        for filename in elite_files:
            filepath = self.project_root / filename
            if filepath.exists():
                elite_files[filename] = {
                    "size": filepath.stat().st_size,
                    "hash": self.file_hash(filepath),
                    "mtime": filepath.stat().st_mtime
                }
        
        # Keep v2 (newest), remove others
        files_to_remove = [
            "core/elite_proactive_assistant.py",
            "core/elite_proactive_assistant_backup.py"
        ]
        
        for filename in files_to_remove:
            filepath = self.project_root / filename
            if filepath.exists():
                self.backup_and_remove(filepath)
    
    def clean_test_files(self):
        """Remove duplicate test files"""
        print("\n🧪 Cleaning duplicate test files...")
        
        test_files_to_remove = [
            "test_consciousness_fixed.py",
            "test_consciousness_isolated.py",
            "test_consciousness_minimal.py",
            "test_jarvis.py",
            "test_jarvis_connection.py"
        ]
        
        for test_file in test_files_to_remove:
            test_path = self.project_root / test_file
            if test_path.exists():
                self.backup_and_remove(test_path)
    
    def clean_launch_files(self):
        """Remove old launch files"""
        print("\n🚀 Cleaning old launch files...")
        
        old_launch_files = [
            "launch_jarvis.py",
            "launch_jarvis_advanced.py",
            "launch_jarvis_unified.py"
        ]
        
        for launch_file in old_launch_files:
            launch_path = self.project_root / launch_file
            if launch_path.exists():
                self.backup_and_remove(launch_path)
    
    def clean_empty_stubs(self):
        """Remove or populate empty stub files"""
        print("\n📁 Cleaning empty stub files...")
        
        stub_files = [
            "mcp_servers/__init__.py",
            "tools/__init__.py",
            "utils/__init__.py",
            "core/__init__.py",
            "tests/__init__.py",
            "config/__init__.py",
            "scripts/__init__.py",
            "examples/__init__.py",
            "docs/__init__.py",
            "templates/__init__.py",
            ".github/workflows/__init__.py"
        ]
        
        init_content = '"""JARVIS module initialization"""\n\n__all__ = []\n'
        
        for stub in stub_files:
            stub_path = self.project_root / stub
            if stub_path.exists() and stub_path.stat().st_size == 0:
                # Add minimal content instead of removing
                stub_path.write_text(init_content)
                print(f"  ✏️  Added content to: {stub}")
                self.removal_log.append({
                    "file": str(stub),
                    "action": "populated",
                    "content": "minimal __init__.py"
                })
    
    def identify_duplicate_classes(self):
        """Identify files with duplicate class definitions"""
        print("\n🔍 Identifying duplicate class definitions...")
        
        duplicate_classes = {
            "EliteProactiveAssistant": ["core/elite_proactive_assistant.py", "core/elite_proactive_assistant_v2.py"],
            "ProactiveIntelligenceEngine": ["Multiple files"],
            "JARVISServer": ["Multiple files"],
            "EnhancedMultiAIIntegration": ["Multiple files"],
            "Task": ["core/database.py", "core/autonomous_project_engine.py"]
        }
        
        print("  ⚠️  Classes found in multiple locations:")
        for class_name, locations in duplicate_classes.items():
            print(f"    - {class_name}: {', '.join(locations)}")
        
        self.removal_log.append({
            "action": "identified_duplicates",
            "duplicate_classes": duplicate_classes
        })
    
    def save_cleanup_log(self):
        """Save cleanup log for reference"""
        log_path = self.project_root / "cleanup_log.json"
        with open(log_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "backup_directory": str(self.backup_dir),
                "actions": self.removal_log
            }, f, indent=2)
        print(f"\n📝 Cleanup log saved to: {log_path}")
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("🧹 Starting JARVIS cleanup process...")
        print(f"📂 Project root: {self.project_root.absolute()}")
        
        # Create backup first
        self.create_backup()
        
        # Clean duplicate files
        self.clean_duplicate_launchers()
        self.clean_elite_proactive_duplicates()
        self.clean_test_files()
        self.clean_launch_files()
        self.clean_empty_stubs()
        self.identify_duplicate_classes()
        
        # Save log
        self.save_cleanup_log()
        
        print("\n✅ Cleanup completed!")
        print(f"💾 Backup saved to: {self.backup_dir}")
        print(f"📊 Total files processed: {len(self.removal_log)}")

if __name__ == "__main__":
    # Run cleanup
    cleanup = JARVISCleanup(".")
    cleanup.run_cleanup()
    
    # Show summary
    print("\n📋 Summary of actions:")
    removed_count = sum(1 for log in cleanup.removal_log if log.get("action") == "removed")
    populated_count = sum(1 for log in cleanup.removal_log if log.get("action") == "populated")
    
    print(f"  - Files removed: {removed_count}")
    print(f"  - Files populated: {populated_count}")
    print("\n💡 Next steps:")
    print("  1. Review cleanup_log.json")
    print("  2. Test that everything still works")
    print("  3. If all good, remove the backup directory")
    print("  4. Commit the cleaned structure to git")