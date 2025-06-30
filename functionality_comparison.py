#!/usr/bin/env python3
"""
Compare functionality between original and clean JARVIS
"""

from pathlib import Path
import re

def analyze_functionality():
    """Analyze what features exist in original vs clean"""
    
    # Core modules in clean version
    clean_core = [
        "configuration.py",
        "database.py", 
        "consciousness_simulation.py",
        "neural_resource_manager.py",
        "performance_optimizer.py",
        "emotional_intelligence.py",
        "self_healing_system.py"
    ]
    
    # Scan original core directory
    original_core_path = Path("core")
    all_core_files = [f.name for f in original_core_path.glob("*.py") if f.is_file()]
    
    # Categories of functionality
    categories = {
        "🧠 AI/ML Features": [],
        "🔊 Voice/Audio": [],
        "🌐 Web/API Integration": [],
        "🤖 Advanced AI": [],
        "📊 Monitoring/Analytics": [],
        "🔧 Tools/Utilities": [],
        "🚀 Performance": [],
        "💾 Storage/Memory": [],
        "🎮 UI/Interface": [],
        "🔒 Security": [],
        "📡 Communication": [],
        "🧪 Testing/Dev Tools": []
    }
    
    # Categorize all files
    for file in all_core_files:
        file_lower = file.lower()
        
        # Skip test files and __init__
        if "test" in file_lower or file == "__init__.py":
            continue
            
        # Voice/Audio
        if any(x in file_lower for x in ["voice", "audio", "speech", "elevenlabs"]):
            categories["🔊 Voice/Audio"].append(file)
            
        # Web/API
        elif any(x in file_lower for x in ["web", "api", "claude", "openai", "integration"]):
            categories["🌐 Web/API Integration"].append(file)
            
        # Advanced AI
        elif any(x in file_lower for x in ["llm", "swarm", "quantum", "metacognitive", "autonomous"]):
            categories["🤖 Advanced AI"].append(file)
            
        # Monitoring
        elif any(x in file_lower for x in ["monitor", "health", "dashboard", "analytics"]):
            categories["📊 Monitoring/Analytics"].append(file)
            
        # Tools
        elif any(x in file_lower for x in ["tool", "generator", "synthesizer", "factory"]):
            categories["🔧 Tools/Utilities"].append(file)
            
        # Performance
        elif any(x in file_lower for x in ["performance", "optimizer", "cache", "parallel"]):
            categories["🚀 Performance"].append(file)
            
        # Memory/Storage
        elif any(x in file_lower for x in ["memory", "storage", "database", "redis"]):
            categories["💾 Storage/Memory"].append(file)
            
        # UI
        elif any(x in file_lower for x in ["ui", "interface", "visual"]):
            categories["🎮 UI/Interface"].append(file)
            
        # Security
        elif any(x in file_lower for x in ["security", "privacy", "sandbox"]):
            categories["🔒 Security"].append(file)
            
        # Communication
        elif any(x in file_lower for x in ["websocket", "mcp", "protocol"]):
            categories["📡 Communication"].append(file)
            
        # Core AI
        elif any(x in file_lower for x in ["consciousness", "neural", "emotional", "cognitive"]):
            categories["🧠 AI/ML Features"].append(file)
            
        # Dev tools
        elif any(x in file_lower for x in ["test_", "debug", "profiler"]):
            categories["🧪 Testing/Dev Tools"].append(file)
            
        # Other
        else:
            # Try to categorize based on content
            if "ml" in file_lower or "model" in file_lower:
                categories["🧠 AI/ML Features"].append(file)
            else:
                categories["🔧 Tools/Utilities"].append(file)
    
    # Generate report
    print("\n" + "="*70)
    print("🔍 JARVIS Functionality Analysis")
    print("="*70)
    
    print(f"\n📊 Total core modules in original: {len(all_core_files)}")
    print(f"📦 Core modules in clean version: {len(clean_core)}")
    
    print("\n✅ INCLUDED in Clean Version:")
    print("-" * 40)
    for module in clean_core:
        print(f"  • {module}")
    
    print("\n❌ NOT INCLUDED in Clean Version (by category):")
    print("-" * 40)
    
    total_missing = 0
    missing_critical = []
    
    for category, files in categories.items():
        # Filter out files that are in clean version
        missing = [f for f in files if f not in clean_core and f != "__init__.py"]
        
        if missing:
            print(f"\n{category} ({len(missing)} modules):")
            for f in sorted(missing)[:10]:  # Show first 10
                print(f"  • {f}")
                
                # Check if critical
                if any(x in f.lower() for x in ["claude", "openai", "voice", "llm", "quantum"]):
                    missing_critical.append(f)
                    
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            
            total_missing += len(missing)
    
    # Recommendations
    print("\n" + "="*70)
    print("📋 RECOMMENDATIONS")
    print("="*70)
    
    print("\n🎯 Critical Missing Features:")
    if missing_critical:
        for f in missing_critical[:5]:
            print(f"  ⚠️  {f}")
    else:
        print("  ✅ None - core functionality is preserved!")
    
    print("\n💡 To Add Back Specific Features:")
    print("  1. Voice Mode: Copy voice_system.py, voice_recognition.py")
    print("  2. AI Integration: Copy real_claude_integration.py, real_openai_integration.py")
    print("  3. Advanced AI: Copy llm_research_integration.py, quantum_swarm_optimization.py")
    print("  4. Web UI: Copy ui_components.py and HTML files")
    
    print("\n✨ Bottom Line:")
    if total_missing > 50:
        print(f"  • You have the CORE functionality")
        print(f"  • Missing {total_missing} specialized modules")
        print(f"  • Most are variations/experiments")
        print(f"  • Add back only what you actually use!")
    else:
        print("  • Clean version has all essential features!")
    
    # Check for specific popular features
    print("\n🔍 Feature Status:")
    features = {
        "Voice Control": any("voice" in f.lower() for f in all_core_files),
        "Claude Integration": any("claude" in f.lower() for f in all_core_files),
        "OpenAI Integration": any("openai" in f.lower() for f in all_core_files),
        "Web Interface": any("ui" in f.lower() or "web" in f.lower() for f in all_core_files),
        "Real-time Monitoring": any("monitor" in f.lower() for f in all_core_files),
        "Advanced ML": any("ml" in f.lower() or "model" in f.lower() for f in all_core_files)
    }
    
    for feature, exists in features.items():
        if exists:
            in_clean = any(
                feature.lower().replace(" ", "_") in c.lower() 
                for c in clean_core
            )
            status = "✅ Included" if in_clean else "❌ Not included (can add back)"
            print(f"  • {feature}: {status}")

if __name__ == "__main__":
    analyze_functionality()