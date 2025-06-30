#!/usr/bin/env python3
"""
Simplified Enhanced Claude Memory RAG Server
Works with minimal dependencies
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib


# Base MCP Server implementation
class SimpleMemoryRAG:
    """Simple memory system with basic functionality"""

    def __init__(self):
        # Local storage
        self.local_dir = Path.home() / ".claude_simple_memory"
        self.local_dir.mkdir(exist_ok=True)

        # Memory storage
        self.memories = {}
        self.conversations = {}
        self.patterns = {}

        self._load_memory()

    def _load_memory(self):
        """Load memory from disk"""
        memory_file = self.local_dir / "memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, "r") as f:
                    data = json.load(f)
                    self.memories = data.get("memories", {})
                    self.conversations = data.get("conversations", {})
                    self.patterns = data.get("patterns", {})
            except Exception as e:
                print(f"Error loading memory: {e}", file=sys.stderr)

    def _save_memory(self):
        """Save memory to disk"""
        memory_file = self.local_dir / "memory.json"
        try:
            with open(memory_file, "w") as f:
                json.dump(
                    {
                        "memories": self.memories,
                        "conversations": self.conversations,
                        "patterns": self.patterns,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Error saving memory: {e}", file=sys.stderr)

    def store_conversation(
        self,
        conversation_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store a conversation"""
        try:
            self.conversations[conversation_id] = {
                "messages": messages,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }

            # Extract key content for memory
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    memory_id = hashlib.md5(
                        f"{content}{datetime.now()}".encode()
                    ).hexdigest()[:8]
                    self.memories[memory_id] = {
                        "content": content,
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                    }

            self._save_memory()
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}", file=sys.stderr)
            return False

    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Simple keyword search in memories"""
        results = []
        query_lower = query.lower()

        for memory_id, memory in self.memories.items():
            content = memory.get("content", "").lower()
            if query_lower in content:
                results.append(
                    {
                        "id": memory_id,
                        "content": memory["content"],
                        "timestamp": memory.get("timestamp"),
                        "relevance": content.count(query_lower) / len(content.split()),
                    }
                )

        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_memories": len(self.memories),
            "total_conversations": len(self.conversations),
            "total_patterns": len(self.patterns),
            "storage_path": str(self.local_dir),
        }

    def learn_pattern(
        self, pattern: str, success: bool, context: Optional[Dict] = None
    ):
        """Store a learned pattern"""
        pattern_id = f"pattern_{datetime.now().timestamp()}"
        self.patterns[pattern_id] = {
            "pattern": pattern,
            "success": success,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._save_memory()


# MCP Server
class SimpleMCPServer:
    """Simple MCP Server implementation"""

    def __init__(self):
        self.memory = SimpleMemoryRAG()
        self.start_time = datetime.now()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests with proper JSON-RPC 2.0 format"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Base response with JSON-RPC 2.0 format
        response = {"jsonrpc": "2.0", "id": request_id}

        try:
            if method == "initialize":
                response["result"] = {
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "simple-claude-memory", "version": "1.0.0"},
                }

            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "store_conversation",
                            "description": "Store a conversation in memory",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "conversation_id": {"type": "string"},
                                    "messages": {"type": "array"},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["conversation_id", "messages"],
                            },
                        },
                        {
                            "name": "search_memories",
                            "description": "Search memories by keyword",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "limit": {"type": "integer", "default": 5},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "get_memory_stats",
                            "description": "Get memory system statistics",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "learn_pattern",
                            "description": "Store a learned pattern",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                    "success": {"type": "boolean"},
                                    "context": {"type": "object"},
                                },
                                "required": ["pattern", "success"],
                            },
                        },
                    ]
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                if tool_name == "store_conversation":
                    success = self.memory.store_conversation(
                        tool_params.get("conversation_id"),
                        tool_params.get("messages", []),
                        tool_params.get("metadata"),
                    )
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Conversation stored successfully: {success}",
                            }
                        ]
                    }

                elif tool_name == "search_memories":
                    memories = self.memory.search_memories(
                        tool_params.get("query", ""), tool_params.get("limit", 5)
                    )
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": json.dumps(memories, indent=2)}
                        ]
                    }

                elif tool_name == "get_memory_stats":
                    stats = self.memory.get_stats()
                    stats["uptime"] = str(datetime.now() - self.start_time)
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": json.dumps(stats, indent=2)}
                        ]
                    }

                elif tool_name == "learn_pattern":
                    self.memory.learn_pattern(
                        tool_params.get("pattern"),
                        tool_params.get("success", True),
                        tool_params.get("context"),
                    )
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": "Pattern learned successfully"}
                        ]
                    }

                else:
                    response["error"] = {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    }

            else:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                }

        except Exception as e:
            response["error"] = {"code": -32603, "message": f"Internal error: {str(e)}"}

        return response

    async def start_server(self):
        """Start the MCP server"""
        print("🧠 Simple Claude Memory Server", file=sys.stderr)
        print("💾 Storage:", self.memory.local_dir, file=sys.stderr)
        print("✅ No external dependencies required!", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("Ready for connections...", file=sys.stderr)

        while True:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break

                request = json.loads(line.strip())

                # Handle request
                response = await self.handle_request(request)

                # Send response to stdout
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                    "id": None,
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except Exception as e:
                print(f"Server error: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Start server
    server = SimpleMCPServer()
    asyncio.run(server.start_server())
