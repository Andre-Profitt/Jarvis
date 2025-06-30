#!/usr/bin/env python3
"""
Master Setup Script - Import JARVIS Knowledge and Setup Project Silos
Runs all imports and configures the project-based memory system
"""

import asyncio
import sys
from pathlib import Path

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from import_jarvis_knowledge import JARVISKnowledgeImporter
from analyze_terminal_history import TerminalHistoryAnalyzer
from project_memory_silos import ProjectMemorySilo


async def main():
    print("🚀 JARVIS Memory System - Complete Setup")
    print("=" * 60)
    print("This will:")
    print("1. Import all JARVIS project knowledge")
    print("2. Analyze terminal history by project")
    print("3. Set up project-based memory silos")
    print("4. Migrate existing memories to silos")
    print("=" * 60)

    response = input("\n🤔 Do you want to proceed? (yes/no): ").lower().strip()

    if response != "yes":
        print("❌ Setup cancelled")
        return

    print("\n" + "=" * 60)

    # Step 1: Import JARVIS knowledge
    print("\n📚 Step 1: Importing JARVIS Knowledge...")
    print("-" * 40)
    try:
        jarvis_importer = JARVISKnowledgeImporter()
        jarvis_stats = await jarvis_importer.run_import()
        print(
            f"✅ JARVIS import complete: {jarvis_stats['memories_created']} memories created"
        )
    except Exception as e:
        print(f"⚠️  JARVIS import error: {e}")

    # Step 2: Analyze terminal history
    print("\n\n🖥️  Step 2: Analyzing Terminal History...")
    print("-" * 40)
    try:
        history_analyzer = TerminalHistoryAnalyzer()
        await history_analyzer.run_analysis()
        print("✅ Terminal history analysis complete")
    except Exception as e:
        print(f"⚠️  Terminal history error: {e}")

    # Step 3: Set up project silos
    print("\n\n📁 Step 3: Setting Up Project Silos...")
    print("-" * 40)
    try:
        silo_system = ProjectMemorySilo()

        # Migrate existing memories
        await silo_system.migrate_existing_memories()

        # Get stats
        all_stats = silo_system.get_all_project_stats()

        print("\n📊 Project Silo Summary:")
        for project_id, stats in all_stats.items():
            print(f"\n{project_id}:")
            print(f"  Description: {stats['description']}")
            print(f"  Total memories: {stats['total_memories']}")

        print("\n✅ Project silos configured")

    except Exception as e:
        print(f"⚠️  Project silo error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ JARVIS Memory System Setup Complete!")
    print("=" * 60)

    print("\n📋 What's been set up:")
    print("  • All JARVIS knowledge imported")
    print("  • Terminal history analyzed by project")
    print("  • Project-based memory silos created")
    print("  • Existing memories migrated to silos")

    print("\n🚀 Next Steps:")
    print("1. Update Claude config to use the project silo server:")
    print("   server: project_memory_silos.py")
    print("2. Restart Claude Desktop")
    print("3. Use set_project_context to switch between projects")

    print("\n💡 Available projects:")
    print("  • JARVIS - Main AI system")
    print("  • claude-memory - Memory RAG system")
    print("  • browser-control - Browser automation")
    print("  • desktop-automation - Desktop control")


if __name__ == "__main__":
    asyncio.run(main())
