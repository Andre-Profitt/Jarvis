#!/usr/bin/env python3
"""
Verify Claude Memory RAG Setup
"""

import os
import sys
import json
import subprocess
from pathlib import Path

print("🔍 Verifying Claude Memory RAG Setup")
print("=" * 60)

# Check paths
server_path = Path(
    "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/mcp_servers/claude-memory-rag/server_simple_working.py"
)
config_path = (
    Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
)
memory_dir = Path.home() / ".claude_simple_memory"

# 1. Check if server exists and is executable
print("\n1️⃣ Checking server file...")
if server_path.exists():
    print(f"✅ Server found: {server_path}")
    if os.access(server_path, os.X_OK):
        print("✅ Server is executable")
    else:
        print("⚠️  Making server executable...")
        os.chmod(server_path, 0o755)
        print("✅ Server is now executable")
else:
    print(f"❌ Server not found at {server_path}")
    sys.exit(1)

# 2. Check Claude configuration
print("\n2️⃣ Checking Claude configuration...")
if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)

    if "claude-memory" in config.get("mcpServers", {}):
        memory_config = config["mcpServers"]["claude-memory"]
        print("✅ Memory server configured in Claude")
        print(f"   Command: {memory_config.get('command')}")
        print(f"   Args: {memory_config.get('args')}")
    else:
        print("❌ Memory server not found in Claude config")
        print("   Run: python3 fix_all_issues.py")
else:
    print(f"❌ Claude config not found at {config_path}")

# 3. Test server protocol
print("\n3️⃣ Testing server protocol...")
try:
    # Test initialize
    result = subprocess.run(
        [sys.executable, str(server_path)],
        input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n',
        capture_output=True,
        text=True,
        timeout=5,
    )

    if result.returncode == 0 and result.stdout:
        response = json.loads(result.stdout.strip())
        if (
            response.get("result", {}).get("serverInfo", {}).get("name")
            == "claude-memory-rag"
        ):
            print("✅ Server responds correctly to initialize")
        else:
            print("⚠️  Server response unexpected")
            print(f"   Response: {response}")
    else:
        print("❌ Server failed to respond")
        if result.stderr:
            print(f"   Error: {result.stderr}")
except Exception as e:
    print(f"❌ Error testing server: {e}")

# 4. Check memory directory
print("\n4️⃣ Checking memory storage...")
if memory_dir.exists():
    print(f"✅ Memory directory exists: {memory_dir}")
    memory_file = memory_dir / "memory.json"
    if memory_file.exists():
        with open(memory_file, "r") as f:
            data = json.load(f)
        print(f"   Stored conversations: {len(data.get('conversations', {}))}")
        print(f"   Stored patterns: {len(data.get('patterns', {}))}")
else:
    print(f"ℹ️  Memory directory will be created on first use: {memory_dir}")

# 5. Check Python dependencies
print("\n5️⃣ Checking Python dependencies...")
required = ["numpy"]
missing = []

for module in required:
    try:
        __import__(module)
        print(f"✅ {module} installed")
    except ImportError:
        print(f"❌ {module} missing")
        missing.append(module)

if missing:
    print(f"\n⚠️  Install missing dependencies:")
    print(f"   {sys.executable} -m pip install {' '.join(missing)}")

# 6. Test tool functionality
print("\n6️⃣ Testing server tools...")
try:
    # Test tools/list
    result = subprocess.run(
        [sys.executable, str(server_path)],
        input='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n',
        capture_output=True,
        text=True,
        timeout=5,
    )

    if result.returncode == 0 and result.stdout:
        response = json.loads(result.stdout.strip())
        tools = response.get("result", {}).get("tools", [])
        print(f"✅ Server provides {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
    else:
        print("❌ Failed to list tools")
except Exception as e:
    print(f"❌ Error testing tools: {e}")

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

all_good = True

if not server_path.exists() or not os.access(server_path, os.X_OK):
    all_good = False
    print("❌ Server file issues")
elif "claude-memory" not in config.get("mcpServers", {}):
    all_good = False
    print("❌ Claude configuration issues")
elif missing:
    all_good = False
    print("❌ Missing dependencies")
else:
    print("✅ All checks passed!")

print("\n🚀 Next Steps:")
if all_good:
    print("1. Restart Claude Desktop")
    print("2. Look for 'claude-memory' in your MCP tools")
    print("3. Start using memory features!")
else:
    print("1. Fix the issues above")
    print("2. Re-run this verification script")
    print("3. Then restart Claude Desktop")

print("\n💡 Quick Test Commands:")
print("- Store a memory: Use the 'store_conversation' tool")
print("- Search memories: Use the 'recall_memories' tool")
print("- Check stats: Use the 'get_memory_stats' tool")
