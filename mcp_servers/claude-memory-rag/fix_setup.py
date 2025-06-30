#!/usr/bin/env python3
"""Fix setup issues for Claude Memory RAG"""
import subprocess
import sys
import os


def run_command(cmd, check=False):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd.split()[0]} succeeded")
            return True
        else:
            print(f"❌ {cmd.split()[0]} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {cmd}: {e}")
        return False


def main():
    print("🔧 Fixing Claude Memory RAG Setup Issues")
    print("=" * 60)

    # 1. Fix sentence-transformers
    print("\n📦 Installing sentence-transformers...")
    if not run_command("pip install sentence-transformers==3.0.1"):
        print("   Trying alternative install...")
        run_command("pip install sentence-transformers --no-deps")
        run_command("pip install transformers torch numpy scikit-learn scipy nltk tqdm")

    # 2. Install faiss
    print("\n📦 Installing faiss...")
    if not run_command("conda install -c pytorch faiss-cpu -y"):
        print("   Trying pip install...")
        run_command("pip install faiss-cpu")

    # 3. Check for OpenAI API key
    print("\n🔑 Checking OpenAI API key...")
    if not os.environ.get("OPENAI_API_KEY"):
        print("   ⚠️  No OPENAI_API_KEY found!")
        print("   Please add to your .env file:")
        print('   OPENAI_API_KEY="your-key-here"')
    else:
        print("   ✅ OpenAI API key found")

    # 4. Create GCS bucket
    print("\n☁️  Creating GCS bucket...")
    if run_command("gsutil mb gs://jarvis-memory-storage 2>/dev/null"):
        print("   ✅ Bucket created")
    else:
        print("   ℹ️  Bucket might already exist or gsutil not configured")
        print("   You can create it manually in Google Cloud Console")

    # 5. Update deprecated packages
    print("\n📦 Updating LangChain packages...")
    run_command("pip install langchain-community langchain-openai --upgrade")

    # 6. Final test
    print("\n🧪 Testing imports...")
    try:
        import sentence_transformers

        print("   ✅ sentence_transformers imports successfully")
    except:
        print("   ❌ sentence_transformers still not working")

    try:
        import faiss

        print("   ✅ faiss imports successfully")
    except:
        print("   ❌ faiss still not working")

    print("\n" + "=" * 60)
    print("🎉 Setup fixes complete!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Create GCS bucket if needed")
    print("3. Run: python3 test_enhanced.py")


if __name__ == "__main__":
    main()
