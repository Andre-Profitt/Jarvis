#!/usr/bin/env python3
"""Quick import check for Claude Memory RAG"""
import sys

print("🧪 Quick Import Check")
print("=" * 40)

modules = [
    ("sentence_transformers", "Sentence Transformers"),
    ("chromadb", "ChromaDB"),
    ("google.cloud.storage", "Google Cloud Storage"),
    ("mem0ai", "Mem0 AI"),
    ("langchain", "LangChain"),
    ("openai", "OpenAI"),
    ("numpy", "NumPy"),
    ("faiss", "FAISS (optional)"),
]

working = []
missing = []

for module, name in modules:
    try:
        __import__(module)
        print(f"✅ {name}")
        working.append(name)
    except ImportError:
        print(f"❌ {name}")
        missing.append(name)

print("\n" + "=" * 40)
print(f"✅ Working: {len(working)}/{len(modules)}")
print(f"❌ Missing: {', '.join(missing) if missing else 'None'}")

if "Sentence Transformers" in working and "Google Cloud Storage" in working:
    print("\n✨ Core components are ready!")
    print("The system can run with local embeddings.")
