#!/usr/bin/env python3
"""
Mem0 MCP Server - Advanced memory for Claude
Mem0 is specifically designed for LLM memory management
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Install: pip install mem0ai


class Mem0MCPServer:
    """MCP Server using Mem0 for advanced memory management"""

    def __init__(self):
        try:
            from mem0 import Memory

            self.memory = Memory()
            self.available = True
            print("✅ Mem0 initialized successfully")
        except ImportError:
            print("⚠️  Mem0 not installed. Install with: pip install mem0ai")
            self.available = False

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        if not self.available:
            return {"error": "Mem0 not available. Install with: pip install mem0ai"}

        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "add_memory":
                # Add a memory
                result = self.memory.add(
                    params.get("content"),
                    user_id=params.get("user_id", "default"),
                    metadata=params.get("metadata", {}),
                )
                return {"result": {"success": True, "memory_id": str(result)}}

            elif method == "search_memory":
                # Search memories
                results = self.memory.search(
                    query=params.get("query"),
                    user_id=params.get("user_id", "default"),
                    limit=params.get("limit", 5),
                )
                return {"result": {"memories": results}}

            elif method == "get_all_memories":
                # Get all memories for a user
                memories = self.memory.get_all(user_id=params.get("user_id", "default"))
                return {"result": {"memories": memories}}

            elif method == "update_memory":
                # Update existing memory
                self.memory.update(
                    memory_id=params.get("memory_id"), content=params.get("content")
                )
                return {"result": {"success": True}}

            elif method == "delete_memory":
                # Delete memory
                self.memory.delete(memory_id=params.get("memory_id"))
                return {"result": {"success": True}}

            elif method == "get_history":
                # Get memory history
                history = self.memory.history(user_id=params.get("user_id", "default"))
                return {"result": {"history": history}}

            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            return {"error": str(e)}

    async def start_server(self):
        """Start the MCP server"""
        print("🧠 Mem0 MCP Server starting...")
        print(
            "📝 Features: Auto-summarization, Entity extraction, User-specific memory"
        )

        while True:
            try:
                # Read request from stdin (MCP protocol)
                line = await asyncio.get_event_loop().run_in_executor(None, input)
                request = json.loads(line)

                # Handle request
                response = await self.handle_request(request)

                # Send response
                print(json.dumps(response))

            except Exception as e:
                print(json.dumps({"error": str(e)}))


# Setup script
def setup_mem0():
    """Setup Mem0 for Claude Desktop"""
    print("🚀 Setting up Mem0 Memory for Claude")
    print("=" * 50)

    # Install Mem0
    import subprocess

    print("📦 Installing Mem0...")
    result = subprocess.run(
        ["pip", "install", "mem0ai"], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("✅ Mem0 installed successfully")
    else:
        print("⚠️  Issue installing Mem0")
        print(result.stderr)

    # Update Claude config
    config_path = (
        Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    )

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}

    # Add Mem0 server
    config["mcpServers"]["claude-mem0"] = {
        "command": "python3",
        "args": [str(Path(__file__))],
        "env": {},
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Claude Desktop configured for Mem0")
    print("\n📝 Next steps:")
    print("1. Restart Claude Desktop")
    print("2. Mem0 will automatically manage your conversation memory")
    print("\n✨ Features you get:")
    print("- Automatic summarization of long conversations")
    print("- Entity and fact extraction")
    print("- User-specific memory profiles")
    print("- Semantic search across all memories")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_mem0()
    else:
        # Run as MCP server
        server = Mem0MCPServer()
        asyncio.run(server.start_server())
