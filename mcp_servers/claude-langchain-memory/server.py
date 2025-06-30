#!/usr/bin/env python3
"""
LangChain Memory MCP Server - Multiple memory types for Claude
Offers conversation buffer, summary, knowledge graph, and vector memory
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import pickle

# Install: pip install langchain chromadb


class LangChainMemoryMCP:
    """MCP Server with multiple LangChain memory types"""

    def __init__(self):
        self.memory_dir = Path.home() / ".claude_langchain_memory"
        self.memory_dir.mkdir(exist_ok=True)

        try:
            from langchain.memory import (
                ConversationBufferMemory,
                ConversationSummaryMemory,
                ConversationKGMemory,
                CombinedMemory,
            )
            from langchain.schema import BaseMessage, HumanMessage, AIMessage

            # Initialize different memory types
            self.buffer_memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

            # For summary memory, we'd need an LLM
            # For now, we'll use buffer memory as primary

            self.memories = {"buffer": self.buffer_memory}

            self.available = True
            self._load_memories()
            print("✅ LangChain memory initialized")

        except ImportError as e:
            print(f"⚠️  LangChain not fully installed: {e}")
            print("Install with: pip install langchain chromadb")
            self.available = False

    def _save_memories(self):
        """Persist memories to disk"""
        for name, memory in self.memories.items():
            try:
                # Save memory state
                memory_file = self.memory_dir / f"{name}_memory.pkl"
                with open(memory_file, "wb") as f:
                    pickle.dump(memory.chat_memory.messages, f)
            except Exception as e:
                print(f"Error saving {name} memory: {e}")

    def _load_memories(self):
        """Load memories from disk"""
        for name, memory in self.memories.items():
            try:
                memory_file = self.memory_dir / f"{name}_memory.pkl"
                if memory_file.exists():
                    with open(memory_file, "rb") as f:
                        messages = pickle.load(f)
                        # Restore messages
                        for msg in messages:
                            memory.chat_memory.messages.append(msg)
            except Exception as e:
                print(f"Error loading {name} memory: {e}")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        if not self.available:
            return {
                "error": "LangChain not available. Install with: pip install langchain"
            }

        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "add_message":
                # Add a message to memory
                memory_type = params.get("memory_type", "buffer")
                role = params.get("role", "human")
                content = params.get("content", "")

                if memory_type in self.memories:
                    memory = self.memories[memory_type]

                    # Add message based on role
                    if role == "human":
                        memory.chat_memory.add_user_message(content)
                    else:
                        memory.chat_memory.add_ai_message(content)

                    self._save_memories()
                    return {"result": {"success": True}}
                else:
                    return {"error": f"Unknown memory type: {memory_type}"}

            elif method == "get_messages":
                # Get conversation history
                memory_type = params.get("memory_type", "buffer")
                limit = params.get("limit", 50)

                if memory_type in self.memories:
                    memory = self.memories[memory_type]
                    messages = memory.chat_memory.messages[-limit:]

                    # Convert to serializable format
                    msg_list = []
                    for msg in messages:
                        msg_list.append(
                            {
                                "role": (
                                    "human"
                                    if msg.__class__.__name__ == "HumanMessage"
                                    else "ai"
                                ),
                                "content": msg.content,
                            }
                        )

                    return {"result": {"messages": msg_list}}
                else:
                    return {"error": f"Unknown memory type: {memory_type}"}

            elif method == "clear_memory":
                # Clear memory
                memory_type = params.get("memory_type", "buffer")

                if memory_type in self.memories:
                    self.memories[memory_type].clear()
                    self._save_memories()
                    return {"result": {"success": True}}
                else:
                    return {"error": f"Unknown memory type: {memory_type}"}

            elif method == "search_memory":
                # Search through memories (basic implementation)
                query = params.get("query", "")
                memory_type = params.get("memory_type", "buffer")

                if memory_type in self.memories:
                    memory = self.memories[memory_type]
                    messages = memory.chat_memory.messages

                    # Simple keyword search
                    results = []
                    for i, msg in enumerate(messages):
                        if query.lower() in msg.content.lower():
                            results.append(
                                {
                                    "index": i,
                                    "role": (
                                        "human"
                                        if msg.__class__.__name__ == "HumanMessage"
                                        else "ai"
                                    ),
                                    "content": msg.content,
                                    "match": True,
                                }
                            )

                    return {"result": {"matches": results[:10]}}  # Limit to 10 results
                else:
                    return {"error": f"Unknown memory type: {memory_type}"}

            elif method == "get_memory_stats":
                # Get memory statistics
                stats = {}
                for name, memory in self.memories.items():
                    stats[name] = {
                        "message_count": len(memory.chat_memory.messages),
                        "type": memory.__class__.__name__,
                    }

                return {"result": {"stats": stats}}

            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            return {"error": str(e)}

    async def start_server(self):
        """Start the MCP server"""
        print("🔗 LangChain Memory MCP Server starting...")
        print("📝 Memory types available: Buffer (more can be added)")
        print(f"💾 Storage location: {self.memory_dir}")

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


def setup_langchain():
    """Setup LangChain memory for Claude Desktop"""
    import subprocess
    from pathlib import Path

    print("🚀 Setting up LangChain Memory for Claude")
    print("=" * 50)

    # Install dependencies
    print("📦 Installing LangChain...")
    deps = ["langchain", "chromadb", "tiktoken"]

    for dep in deps:
        print(f"   Installing {dep}...")
        result = subprocess.run(["pip", "install", dep], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {dep} installed")
        else:
            print(f"   ⚠️  Issue with {dep}")

    # Update Claude config
    config_path = (
        Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    )

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}

    # Add LangChain server
    config["mcpServers"]["claude-langchain-memory"] = {
        "command": "python3",
        "args": [str(Path(__file__).absolute())],
        "env": {"PYTHONPATH": str(Path(__file__).parent.parent.parent)},
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Claude Desktop configured for LangChain Memory")
    print("\n📝 Next steps:")
    print("1. Restart Claude Desktop")
    print("2. LangChain memory will manage conversations")
    print("\n✨ Features available:")
    print("- Conversation Buffer Memory (full history)")
    print("- Memory search functionality")
    print("- Persistent storage between sessions")
    print("- Easy to extend with more memory types")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_langchain()
    else:
        # Run as MCP server
        server = LangChainMemoryMCP()
        asyncio.run(server.start_server())
