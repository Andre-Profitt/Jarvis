#!/usr/bin/env python3
"""
Update Claude configuration to use project-based memory silos
"""

import json
from pathlib import Path


def update_config():
    """Update Claude config to use project silo server"""
    config_path = (
        Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    )

    print("📝 Updating Claude configuration...")

    # Read current config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Update memory server to use project silos
    config["mcpServers"]["claude-memory"] = {
        "command": "/Users/andreprofitt/opt/anaconda3/bin/python3",
        "args": [
            "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/mcp_servers/claude-memory-rag/project_memory_silos.py",
            "server",  # Important: add server argument
        ],
        "env": {
            "OPENAI_API_KEY": "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",
            "MEM0_API_KEY": "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC",
            "LANGCHAIN_API_KEY": "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",
            "GOOGLE_APPLICATION_CREDENTIALS": "/Users/andreprofitt/.gcs/jarvis-credentials.json",
            "PYTHONPATH": "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM",
        },
    }

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration updated!")
    print("\n📋 New features available:")
    print("  • set_project_context - Switch between projects")
    print("  • store_project_memory - Store with project isolation")
    print("  • search_project_memories - Search within/across projects")
    print("  • get_project_summary - Get project insights")
    print("  • get_all_project_stats - View all project stats")
    print("\n🚀 Restart Claude Desktop to activate!")


if __name__ == "__main__":
    update_config()
