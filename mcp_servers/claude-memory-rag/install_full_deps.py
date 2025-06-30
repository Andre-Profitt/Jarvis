#!/usr/bin/env python3
"""
Install dependencies for full-featured Claude Memory RAG
"""

import subprocess
import sys
import os

print("🔧 Installing dependencies for LangChain + Mem0 + OpenAI")
print("=" * 60)

# Set environment variable for OpenAI
os.environ["OPENAI_API_KEY"] = (
    "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"
)

deps = [
    # Core dependencies
    "openai",  # For OpenAI embeddings
    "mem0ai",  # Mem0 with OpenAI
    "langchain",  # Core LangChain
    "langchain-community",  # Community integrations
    "langchain-openai",  # OpenAI integration for LangChain
    # Vector stores (use conda-friendly alternatives)
    "chromadb",  # Works better than FAISS
    "sentence-transformers",  # For local embeddings backup
    # Already installed
    "google-cloud-storage",
    "numpy",
]

print("\n📦 Installing with pip...")
for dep in deps:
    print(f"Installing {dep}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", dep],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✅ {dep} installed")
    else:
        print(f"⚠️  {dep} had issues: {result.stderr[:100]}")

# Try to install faiss with conda if available
print("\n📦 Trying to install FAISS with conda...")
try:
    result = subprocess.run(
        ["conda", "install", "-c", "conda-forge", "faiss-cpu", "-y"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✅ FAISS installed with conda")
    else:
        print("⚠️  FAISS not available, will use ChromaDB instead")
except:
    print("⚠️  Conda not available, using ChromaDB for vectors")

print("\n✅ Dependencies installed!")
print("\n🔍 Testing imports...")

# Test imports
try:
    import openai

    print("✅ OpenAI imported")

    import mem0

    print("✅ Mem0 imported")

    import langchain

    print("✅ LangChain imported")

    from langchain_openai import OpenAIEmbeddings

    print("✅ LangChain OpenAI imported")

    import chromadb

    print("✅ ChromaDB imported")

    try:
        import faiss

        print("✅ FAISS imported")
        has_faiss = True
    except:
        print("ℹ️  FAISS not available, using ChromaDB")
        has_faiss = False

except ImportError as e:
    print(f"❌ Import error: {e}")

print("\n✅ Setup complete!")
print("Vector store will use:", "FAISS" if has_faiss else "ChromaDB")
