#!/usr/bin/env python3
"""
Setup Claude Memory RAG System
Configures persistent memory for Claude using 30TB Google Cloud Storage
"""

import os
import json
import subprocess
import sys
from pathlib import Path


class ClaudeMemorySetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.claude_config_path = (
            Path.home()
            / "Library/Application Support/Claude/claude_desktop_config.json"
        )

    def install_dependencies(self):
        """Install required Python packages"""
        print("📦 Installing dependencies...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(self.base_dir / "requirements.txt"),
            ],
            check=True,
        )
        print("✅ Dependencies installed!")

    def setup_gcs_credentials(self):
        """Help user set up GCS credentials"""
        print("\n🔑 Setting up Google Cloud Storage credentials...")

        # Check if credentials already exist
        cred_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_env and Path(cred_env).exists():
            print(f"✅ Using existing credentials: {cred_env}")
            return

        print("\n⚠️  Google Cloud credentials not found!")
        print("Please follow these steps:")
        print("1. Go to: https://console.cloud.google.com/iam-admin/serviceaccounts")
        print("2. Create a service account with Storage Admin permissions")
        print("3. Download the JSON key file")
        print("4. Set the environment variable:")
        print("   export GOOGLE_APPLICATION_CREDENTIALS='path/to/your-key.json'")
        print("\nOr place the file at: ~/.gcs/jarvis-credentials.json")

        # Create directory for credentials
        gcs_dir = Path.home() / ".gcs"
        gcs_dir.mkdir(exist_ok=True)

    def update_claude_config(self):
        """Add memory RAG to Claude Desktop config"""
        print("\n🔧 Updating Claude Desktop configuration...")

        # Load existing config or create new
        if self.claude_config_path.exists():
            with open(self.claude_config_path, "r") as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}

        # Add memory RAG server
        config["mcpServers"]["claude-memory-rag"] = {
            "command": "python",
            "args": [str(self.base_dir / "server.py")],
            "env": {
                "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get(
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    str(Path.home() / ".gcs/jarvis-credentials.json"),
                ),
                "GCS_BUCKET": "jarvis-30tb-storage",
            },
        }

        # Save updated config
        self.claude_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.claude_config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Claude Desktop config updated!")

    def create_test_script(self):
        """Create a test script to verify the setup"""
        test_script = '''#!/usr/bin/env python3
"""Test Claude Memory RAG"""

import asyncio
import sys
sys.path.append(str(Path(__file__).parent))

from server import ClaudeMemoryRAG

async def test_memory():
    print("🧪 Testing Claude Memory RAG...")
    
    # Initialize memory
    memory = ClaudeMemoryRAG()
    
    # Test storing a conversation
    print("\\n1️⃣ Testing conversation storage...")
    success = await memory.store_conversation_memory(
        conversation_id="test_001",
        messages=[
            {"role": "user", "content": "How do I implement neural networks?"},
            {"role": "assistant", "content": "Here's how to implement neural networks..."}
        ],
        metadata={"topic": "machine learning"}
    )
    print(f"   Storage result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test recalling memories
    print("\\n2️⃣ Testing memory recall...")
    memories = await memory.recall_relevant_memories(
        query="neural network implementation",
        top_k=3
    )
    print(f"   Found {len(memories)} relevant memories")
    
    # Test code understanding
    print("\\n3️⃣ Testing code understanding storage...")
    success = await memory.store_code_understanding(
        file_path="test_neural.py",
        code_content="class NeuralNetwork:\\n    def __init__(self):\\n        pass",
        analysis="Basic neural network class structure",
        metadata={"language": "python"}
    )
    print(f"   Storage result: {'✅ Success' if success else '❌ Failed'}")
    
    # Get stats
    print("\\n4️⃣ Memory statistics:")
    stats = memory.get_memory_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Memory types: {stats['memory_types']}")
    
    # Sync to GCS
    print("\\n5️⃣ Syncing to Google Cloud Storage...")
    await memory.sync_to_gcs()
    
    print("\\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_memory())
'''

        test_path = self.base_dir / "test_memory.py"
        test_path.write_text(test_script)
        test_path.chmod(0o755)
        print("✅ Test script created: test_memory.py")

    def run_setup(self):
        """Run complete setup"""
        print("🚀 Setting up Claude Memory RAG System")
        print("=" * 50)

        # Install dependencies
        self.install_dependencies()

        # Setup GCS
        self.setup_gcs_credentials()

        # Update Claude config
        self.update_claude_config()

        # Create test script
        self.create_test_script()

        print("\n✨ Setup complete!")
        print("\n📝 Next steps:")
        print("1. Set up GCS credentials if not already done")
        print("2. Restart Claude Desktop to load the new MCP server")
        print("3. Test with: python test_memory.py")
        print("\n🧠 Claude now has persistent memory with your 30TB storage!")


if __name__ == "__main__":
    setup = ClaudeMemorySetup()
    setup.run_setup()
