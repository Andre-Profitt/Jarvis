#!/usr/bin/env python3
"""
Quick fix for the remaining setup
"""

import json
from pathlib import Path

# Update Claude config
config_path = (
    Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
)

if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {"mcpServers": {}}

# Add simple server
config["mcpServers"]["claude-memory-rag-simple"] = {
    "command": "python3",
    "args": [str(Path.cwd() / "server_simple_working.py")],
    "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": str(
            Path.home() / ".gcs/jarvis-credentials.json"
        ),
        "PYTHONPATH": str(Path.cwd().parent.parent),
    },
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Claude config updated!")
print("\n📝 Next steps:")
print("1. Restart Claude Desktop")
print("2. Test with: python3 test_simple_server.py")
