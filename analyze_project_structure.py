#!/usr/bin/env python3
"""
Analyze JARVIS project structure to identify active vs obsolete files
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import re

class ProjectAnalyzer:
    def __init__(self):
        self.root = Path(".")
        self.active_files = set()
        self.test_files = set()
        self.backup_files = set()
        self.launcher_files = set()
        self.phase_files = set()
        self.documentation = set()
        self.core_modules = set()
        self.html_files = set()
        self.config_files = set()
        self.temp_files = set()
        
    def analyze(self):
        """Analyze entire project structure"""
        print("🔍 Analyzing JARVIS Project Structure...\n")
        
        # Categorize all files
        for file_path in self.root.rglob("*"):
            if file_path.is_file() and not str(file_path).startswith(".git"):
                self.categorize_file(file_path)
        
        # Generate report
        self.generate_report()
        
    def categorize_file(self, file_path):
        """Categorize a single file"""
        path_str = str(file_path)
        
        # Skip git files
        if ".git" in path_str:
            return
            
        # Backup/Archive files
        if any(x in path_str for x in [".backup", ".archive", "/backup/", "/archive/", "old_"]):
            self.backup_files.add(file_path)
            
        # Test files
        elif path_str.startswith("tests/") or "test_" in path_str:
            self.test_files.add(file_path)
            
        # Launcher files (we have WAY too many)
        elif any(x in path_str for x in ["launch", "LAUNCH", "start_jarvis", "run_jarvis"]):
            self.launcher_files.add(file_path)
            
        # Phase files (old iterations)
        elif "phase" in path_str.lower():
            self.phase_files.add(file_path)
            
        # Documentation
        elif path_str.endswith(".md") or path_str.endswith(".txt"):
            self.documentation.add(file_path)
            
        # HTML files (UIs)
        elif path_str.endswith(".html"):
            self.html_files.add(file_path)
            
        # Core modules
        elif path_str.startswith("core/") and path_str.endswith(".py"):
            self.core_modules.add(file_path)
            
        # Config files
        elif path_str.startswith("config/") or path_str.endswith((".yaml", ".yml", ".json")):
            self.config_files.add(file_path)
            
        # Temp/generated files
        elif any(x in path_str for x in ["coverage", ".pyc", "__pycache__", ".db", ".log", ".pid"]):
            self.temp_files.add(file_path)
            
        # Active Python files
        elif path_str.endswith(".py"):
            self.active_files.add(file_path)
            
    def generate_report(self):
        """Generate analysis report"""
        report = []
        
        report.append("# 📊 JARVIS Project Analysis Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        report.append("## 📈 Summary\n")
        total_files = sum(len(s) for s in [
            self.active_files, self.test_files, self.backup_files,
            self.launcher_files, self.phase_files, self.documentation,
            self.core_modules, self.html_files, self.config_files, self.temp_files
        ])
        report.append(f"Total files analyzed: {total_files}\n")
        
        # Categories
        categories = [
            ("🗄️ Backup/Archive Files", self.backup_files, "Can be moved to .archive/"),
            ("🚀 Launcher Files", self.launcher_files, "Keep only 1-2 main launchers"),
            ("📝 Phase Files", self.phase_files, "Old iterations - archive"),
            ("🌐 HTML Files", self.html_files, "Multiple UIs - consolidate"),
            ("🧪 Test Files", self.test_files, "Keep all - good coverage!"),
            ("📚 Documentation", self.documentation, "Review and consolidate"),
            ("⚙️ Core Modules", self.core_modules, "Essential - keep"),
            ("📋 Config Files", self.config_files, "Review for duplicates"),
            ("🗑️ Temp Files", self.temp_files, "Safe to delete"),
            ("🐍 Other Python Files", self.active_files, "Review individually")
        ]
        
        for title, file_set, recommendation in categories:
            report.append(f"\n## {title}\n")
            report.append(f"Count: {len(file_set)}\n")
            report.append(f"Recommendation: {recommendation}\n")
            
            if len(file_set) > 0 and len(file_set) < 50:  # Don't list if too many
                report.append("\nFiles:\n")
                for f in sorted(file_set)[:20]:  # Limit to 20
                    report.append(f"- {f}\n")
                if len(file_set) > 20:
                    report.append(f"... and {len(file_set) - 20} more\n")
        
        # Specific recommendations
        report.append("\n## 🎯 Recommended Actions\n")
        report.append("1. **Create Clean Structure**:\n")
        report.append("   ```\n")
        report.append("   JARVIS-CLEAN/\n")
        report.append("   ├── jarvis.py          # Main entry point\n")
        report.append("   ├── core/              # Essential modules only\n")
        report.append("   ├── config/            # Minimal configs\n")
        report.append("   ├── tests/             # All tests (good!)\n")
        report.append("   ├── docs/              # Consolidated docs\n")
        report.append("   ├── requirements.txt   # Dependencies\n")
        report.append("   └── README.md          # Clear instructions\n")
        report.append("   ```\n")
        
        report.append("\n2. **Archive Everything Else**:\n")
        report.append("   - Move to `.archive/` directory\n")
        report.append("   - Keep git history intact\n")
        report.append("   - Can recover if needed\n")
        
        # Duplicate detection
        report.append("\n## 🔄 Potential Duplicates\n")
        
        # Find launcher duplicates
        launcher_patterns = defaultdict(list)
        for f in self.launcher_files:
            base = str(f).lower().replace("_", "").replace("-", "")
            launcher_patterns[base].append(f)
        
        for pattern, files in launcher_patterns.items():
            if len(files) > 1:
                report.append(f"\nLauncher group '{pattern}':\n")
                for f in files:
                    report.append(f"  - {f}\n")
        
        # Save report
        report_path = Path("PROJECT_ANALYSIS_REPORT.md")
        report_path.write_text("".join(report))
        
        # Also print summary
        print("📊 Analysis Complete!\n")
        print(f"Total files: {total_files}")
        print(f"Backup/Archive files: {len(self.backup_files)} (can be archived)")
        print(f"Launcher files: {len(self.launcher_files)} (need only 1-2)")
        print(f"Phase files: {len(self.phase_files)} (old iterations)")
        print(f"Test files: {len(self.test_files)} (keep all!)")
        print(f"Temp files: {len(self.temp_files)} (can be deleted)")
        print("\n✅ Full report saved to: PROJECT_ANALYSIS_REPORT.md")
        
        # Create cleanup script
        self.create_cleanup_script()
        
    def create_cleanup_script(self):
        """Create a cleanup script"""
        script = ['#!/usr/bin/env python3\n']
        script.append('"""\n')
        script.append('Smart cleanup script for JARVIS project\n')
        script.append('"""\n\n')
        script.append('import os\n')
        script.append('import shutil\n')
        script.append('from pathlib import Path\n\n')
        
        script.append('def cleanup():\n')
        script.append('    """Move files to archive"""\n')
        script.append('    archive_dir = Path(".archive")\n')
        script.append('    archive_dir.mkdir(exist_ok=True)\n\n')
        
        # Archive old launchers
        script.append('    # Archive old launchers\n')
        script.append('    launcher_archive = archive_dir / "old_launchers"\n')
        script.append('    launcher_archive.mkdir(exist_ok=True)\n')
        
        for f in self.launcher_files:
            if "jarvis_simple.py" not in str(f):  # Keep the simple one
                script.append(f'    if Path("{f}").exists():\n')
                script.append(f'        shutil.move("{f}", launcher_archive / "{f.name}")\n')
        
        # Archive phase files
        script.append('\n    # Archive phase files\n')
        script.append('    phase_archive = archive_dir / "phase_files"\n')
        script.append('    phase_archive.mkdir(exist_ok=True)\n')
        
        for f in self.phase_files:
            if not str(f).startswith("phase"):  # Don't move directories
                script.append(f'    if Path("{f}").exists():\n')
                script.append(f'        shutil.move("{f}", phase_archive / "{f.name}")\n')
        
        # Delete temp files
        script.append('\n    # Delete temp files\n')
        for f in self.temp_files:
            script.append(f'    if Path("{f}").exists():\n')
            script.append(f'        Path("{f}").unlink()\n')
        
        script.append('\n    print("✅ Cleanup complete!")\n\n')
        script.append('if __name__ == "__main__":\n')
        script.append('    cleanup()\n')
        
        cleanup_path = Path("smart_cleanup.py")
        cleanup_path.write_text("".join(script))
        cleanup_path.chmod(0o755)
        
        print("\n🧹 Cleanup script created: smart_cleanup.py")
        print("   Run with: python smart_cleanup.py")

if __name__ == "__main__":
    analyzer = ProjectAnalyzer()
    analyzer.analyze()