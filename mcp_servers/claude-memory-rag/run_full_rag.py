#!/usr/bin/env python3
"""
Complete RAG Setup and Run Script
This handles everything in one go
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    print("🚀 JARVIS Full RAG System - Complete Setup")
    print("=" * 60)

    # Check current directory
    current_dir = Path.cwd()
    if "claude-memory-rag" not in str(current_dir):
        print("⚠️  Please run from the claude-memory-rag directory:")
        print(
            "   cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/mcp_servers/claude-memory-rag"
        )
        print("   python3 run_full_rag.py")
        return

    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Loading API keys from .env file")
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    # Set Google Cloud credentials
    gcs_creds = Path.home() / ".gcs/jarvis-credentials.json"
    if gcs_creds.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gcs_creds)
        print("✅ Google Cloud credentials found")

    print("\n📊 Configuration:")
    print(f"   Mem0 API: {os.environ.get('MEM0_API_KEY', 'Not set')[:20]}...")
    print(f"   LangChain API: {os.environ.get('LANGCHAIN_API_KEY', 'Not set')[:20]}...")
    print(f"   GCS Credentials: {gcs_creds.exists()}")

    # Ask what to do
    print("\n📋 What would you like to do?")
    print("1. Run complete setup (install deps + configure)")
    print("2. Just test current setup")
    print("3. Start the memory server")
    print("4. Index JARVIS codebase")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        # Run full setup
        print("\n🔧 Running complete setup...")
        subprocess.run([sys.executable, "setup_enhanced_rag.py"])

    elif choice == "2":
        # Test setup
        print("\n🧪 Testing current setup...")
        if Path("test_enhanced.py").exists():
            subprocess.run([sys.executable, "test_enhanced.py"])
        else:
            print("Test script not found. Run setup first (option 1)")

    elif choice == "3":
        # Start server
        print("\n🚀 Starting memory server...")
        if Path("server_enhanced.py").exists():
            subprocess.run([sys.executable, "server_enhanced.py"])
        else:
            print("Server not found. Run setup first (option 1)")

    elif choice == "4":
        # Index JARVIS
        print("\n📚 Indexing JARVIS codebase...")
        if Path("index_jarvis.py").exists():
            subprocess.run([sys.executable, "index_jarvis.py"])
        else:
            print("Indexer not found")

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
