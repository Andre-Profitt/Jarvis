#!/bin/bash
# Smart JARVIS Cleanup - Focus on what matters

echo "🧹 Smart JARVIS Cleanup - Minimal Effort, Maximum Impact"
echo "========================================================"

# 1. Just hide the clutter (don't delete anything!)
echo "📦 Step 1: Archive old files (safe, reversible)"

# Create archive directory
mkdir -p .archive
mkdir -p .archive/old_launchers
mkdir -p .archive/test_files
mkdir -p .archive/phase_files

# Move old launchers (but keep the main ones)
echo "Moving old launcher files..."
for file in launch_jarvis_*.py jarvis_launcher*.py jarvis_*_launcher.py; do
    if [[ -f "$file" && "$file" != "LAUNCH-JARVIS-REAL.py" ]]; then
        mv "$file" .archive/old_launchers/ 2>/dev/null
    fi
done

# Move test files from root (they belong in tests/)
echo "Moving test files to tests directory..."
for file in test_*.py; do
    if [[ -f "$file" ]]; then
        mv "$file" tests/ 2>/dev/null
    fi
done

# Move phase files (old development phases)
echo "Archiving phase files..."
for file in *phase*.py setup_jarvis_phase*.py; do
    if [[ -f "$file" ]]; then
        mv "$file" .archive/phase_files/ 2>/dev/null
    fi
done

# 2. Create a simple launcher that works
echo ""
echo "📝 Step 2: Creating simple unified launcher"

cat > start_jarvis.py << 'EOF'
#!/usr/bin/env python3
"""
Simple JARVIS Starter - Just Works™
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Start JARVIS with the best available configuration"""
    
    print("""
    🤖 JARVIS UNIFIED LAUNCHER
    ==========================
    """)
    
    # Check for main launcher
    if Path("LAUNCH-JARVIS-REAL.py").exists():
        print("Starting JARVIS with full configuration...")
        subprocess.run([sys.executable, "LAUNCH-JARVIS-REAL.py"])
    elif Path("jarvis.py").exists():
        print("Starting JARVIS core...")
        subprocess.run([sys.executable, "jarvis.py"])
    else:
        print("❌ No JARVIS launcher found!")
        print("Looking for alternatives...")
        
        # Find any working JARVIS file
        jarvis_files = list(Path(".").glob("jarvis*.py"))
        if jarvis_files:
            print(f"Found {jarvis_files[0]}, starting...")
            subprocess.run([sys.executable, str(jarvis_files[0])])
        else:
            print("No JARVIS files found. Please check your installation.")

if __name__ == "__main__":
    main()
EOF

chmod +x start_jarvis.py

# 3. Create a clean view symlink structure (optional but nice)
echo ""
echo "🔗 Step 3: Creating clean project view"

# Create a 'clean' directory with symlinks to important files only
mkdir -p jarvis-clean
cd jarvis-clean

# Link only the essential files
ln -sf ../LAUNCH-JARVIS-REAL.py main.py 2>/dev/null
ln -sf ../core core 2>/dev/null
ln -sf ../config config 2>/dev/null
ln -sf ../README.md README.md 2>/dev/null
ln -sf ../requirements.txt requirements.txt 2>/dev/null
ln -sf ../.env .env 2>/dev/null

cd ..

# 4. Create a project status file
echo ""
echo "📊 Step 4: Generating project summary"

cat > PROJECT_STATUS.md << 'EOF'
# JARVIS Project Status

## 🚀 Quick Start
```bash
python start_jarvis.py
```

## 📁 Project Structure
- **Main Files**: Core JARVIS files in root
- **Core Logic**: `./core/` directory
- **Tests**: `./tests/` directory  
- **Archived**: `./.archive/` (old files, safe to ignore)

## 🗂️ What We Did
1. Archived old launcher variants to `.archive/old_launchers/`
2. Moved test files to proper `tests/` directory
3. Archived old phase files to `.archive/phase_files/`
4. Created simple `start_jarvis.py` launcher
5. Created `jarvis-clean/` for a cleaner view (optional)

## ✅ Benefits
- Root directory is now cleaner
- All files are preserved (nothing deleted)
- Can easily restore any file from `.archive/`
- Single entry point: `python start_jarvis.py`
EOF

# 5. Summary
echo ""
echo "✅ CLEANUP COMPLETE!"
echo "==================="
echo ""
echo "What we did:"
echo "- Archived ${ls .archive/old_launchers/*.py 2>/dev/null | wc -l} old launcher files"
echo "- Moved ${ls tests/test_*.py 2>/dev/null | wc -l} test files to tests/"
echo "- Archived ${ls .archive/phase_files/*.py 2>/dev/null | wc -l} phase files"
echo ""
echo "Your root directory is now cleaner!"
echo ""
echo "🚀 To start JARVIS: python start_jarvis.py"
echo "📁 Old files are safe in: .archive/"
echo "🧹 To undo: mv .archive/*/* . "
echo ""
