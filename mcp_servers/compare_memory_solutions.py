#!/usr/bin/env python3
"""
Claude Memory Solutions Comparison & Setup
Choose the best memory system for your needs
"""

import os
import sys
import subprocess
from pathlib import Path


def print_comparison():
    """Show comparison of memory solutions"""
    print(
        """
🧠 CLAUDE MEMORY SOLUTIONS COMPARISON
=====================================

1️⃣  **Our Custom RAG + GCS** (Already Built)
   ✅ Pros: Full control, 30TB storage, custom embeddings
   ❌ Cons: Dependency issues with conda
   📦 Status: Built but needs dependencies fixed
   
2️⃣  **Mem0** (Recommended for Quick Start)
   ✅ Pros: Purpose-built for LLMs, auto-summarization, user profiles
   ❌ Cons: Requires additional install
   📦 Install: pip install mem0ai
   🚀 Best for: Personal AI assistants
   
3️⃣  **LangChain Memory**
   ✅ Pros: Multiple memory types, flexible, well-documented
   ❌ Cons: More complex setup
   📦 Install: pip install langchain
   🚀 Best for: Complex workflows, multiple memory strategies
   
4️⃣  **Simple Local Memory** (Fastest Setup)
   ✅ Pros: No dependencies, works immediately
   ❌ Cons: Basic features only
   📦 Install: Nothing! Already included
   🚀 Best for: Getting started quickly

5️⃣  **Zep** (High Performance)
   ✅ Pros: Very fast, auto-extraction, temporal awareness
   ❌ Cons: Requires separate server
   📦 Install: docker run -p 8000:8000 zepai/zep
   🚀 Best for: Production systems
"""
    )


def setup_simple_memory():
    """Setup the simple memory solution"""
    print("\n🚀 Setting up Simple Memory (no dependencies)...")

    # The simple server is already created
    simple_server = (
        Path(__file__).parent.parent / "claude-memory-rag" / "server_simple.py"
    )

    if simple_server.exists():
        print("✅ Simple memory server found")

        # Run conda setup
        setup_script = simple_server.parent / "conda_setup.py"
        if setup_script.exists():
            subprocess.run([sys.executable, str(setup_script)])
        else:
            print("⚠️  Setup script not found")
    else:
        print("❌ Simple server not found")


def setup_mem0():
    """Setup Mem0 memory"""
    print("\n🚀 Setting up Mem0...")

    # Install mem0
    print("📦 Installing mem0ai...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "mem0ai"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✅ Mem0 installed")

        # Run setup
        mem0_setup = Path(__file__).parent.parent / "claude-mem0" / "server.py"
        if mem0_setup.exists():
            subprocess.run([sys.executable, str(mem0_setup), "setup"])
        else:
            print("⚠️  Mem0 server not found")
    else:
        print("❌ Failed to install Mem0")
        print(result.stderr)


def setup_langchain():
    """Setup LangChain memory"""
    print("\n🚀 Setting up LangChain Memory...")

    # Run setup
    langchain_setup = (
        Path(__file__).parent.parent / "claude-langchain-memory" / "server.py"
    )
    if langchain_setup.exists():
        subprocess.run([sys.executable, str(langchain_setup), "setup"])
    else:
        print("❌ LangChain server not found")


def main():
    """Main setup menu"""
    print_comparison()

    print("\n📋 QUICK SETUP OPTIONS:")
    print("1. Simple Memory (works now, no dependencies)")
    print("2. Mem0 (best features, easy setup)")
    print("3. LangChain (most flexible)")
    print("4. Fix original RAG (if you want the full system)")
    print("5. Compare features again")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        setup_simple_memory()
    elif choice == "2":
        setup_mem0()
    elif choice == "3":
        setup_langchain()
    elif choice == "4":
        print("\n📝 To fix the original RAG:")
        print(
            "1. Use conda to install: conda install -c conda-forge google-cloud-storage"
        )
        print("2. Or create a new virtual environment")
        print("3. Then run: python3 fixed_setup.py")
    elif choice == "5":
        print_comparison()
    else:
        print("❌ Invalid choice")

    print("\n✅ After setup, restart Claude Desktop!")


if __name__ == "__main__":
    main()
