#!/usr/bin/env python3
"""
JARVIS MCP Server - Memory & Storage Integration
==============================================

This MCP server provides Claude Code access to:
- mem0 long-term memory
- LangChain conversation memory
- Google Cloud Storage (30TB)
- JARVIS knowledge base

Run this as an MCP server to give Claude persistent memory and storage access.
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JARVISMemoryStorageMCP:
    """
    MCP Server for JARVIS memory and storage integration
    """

    def __init__(self):
        self.server_info = {
            "name": "jarvis-memory-storage",
            "version": "1.0.0",
            "vendor": "JARVIS-ECOSYSTEM",
            "capabilities": {
                "tools": {"listTools": True, "callTool": True},
                "resources": {"listResources": True, "readResource": True},
                "prompts": {"listPrompts": True, "getPrompt": True},
            },
        }

        # Initialize components (lazy loading)
        self._memory = None
        self._langchain_memory = None
        self._storage_client = None
        self._knowledge_base = {}

        # Define available tools
        self.tools = {
            "store_memory": {
                "description": "Store information in long-term memory",
                "parameters": {
                    "content": {
                        "type": "string",
                        "description": "Content to store",
                        "required": True,
                    },
                    "category": {
                        "type": "string",
                        "description": "Memory category (conversation, knowledge, task, etc.)",
                        "required": False,
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                        "required": False,
                    },
                },
            },
            "retrieve_memory": {
                "description": "Search and retrieve from long-term memory",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                        "required": True,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category",
                        "required": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 10)",
                        "required": False,
                    },
                },
            },
            "store_file": {
                "description": "Store file in Google Cloud Storage",
                "parameters": {
                    "file_path": {
                        "type": "string",
                        "description": "Local file path",
                        "required": True,
                    },
                    "storage_path": {
                        "type": "string",
                        "description": "Path in storage bucket",
                        "required": False,
                    },
                    "metadata": {
                        "type": "object",
                        "description": "File metadata",
                        "required": False,
                    },
                },
            },
            "retrieve_file": {
                "description": "Retrieve file from Google Cloud Storage",
                "parameters": {
                    "storage_path": {
                        "type": "string",
                        "description": "Path in storage bucket",
                        "required": True,
                    },
                    "local_path": {
                        "type": "string",
                        "description": "Local path to save file",
                        "required": False,
                    },
                },
            },
            "list_files": {
                "description": "List files in storage",
                "parameters": {
                    "prefix": {
                        "type": "string",
                        "description": "Path prefix filter",
                        "required": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                        "required": False,
                    },
                },
            },
            "update_knowledge": {
                "description": "Update JARVIS knowledge base",
                "parameters": {
                    "topic": {
                        "type": "string",
                        "description": "Knowledge topic",
                        "required": True,
                    },
                    "content": {
                        "type": "string",
                        "description": "Knowledge content",
                        "required": True,
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of knowledge",
                        "required": False,
                    },
                },
            },
            "search_knowledge": {
                "description": "Search JARVIS knowledge base",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                        "required": True,
                    },
                    "topic": {
                        "type": "string",
                        "description": "Filter by topic",
                        "required": False,
                    },
                },
            },
        }

        # Define available prompts
        self.prompts = {
            "remember_conversation": {
                "description": "Remember this conversation for future reference",
                "arguments": ["conversation_summary"],
            },
            "recall_context": {
                "description": "Recall relevant context from memory",
                "arguments": ["current_topic"],
            },
            "save_learning": {
                "description": "Save what JARVIS has learned",
                "arguments": ["learning_content", "topic"],
            },
        }

    @property
    def memory(self):
        """Lazy load mem0 memory"""
        if self._memory is None:
            try:
                from mem0 import Memory

                self._memory = Memory.from_config(
                    {
                        "vector_store": {
                            "provider": "chroma",
                            "config": {
                                "collection_name": "jarvis_memory",
                                "path": "./storage/chroma",
                            },
                        }
                    }
                )
                logger.info("mem0 memory initialized")
            except ImportError:
                logger.warning("mem0 not available, using simple memory")
                self._memory = SimpleMemory()
        return self._memory

    @property
    def langchain_memory(self):
        """Lazy load LangChain memory"""
        if self._langchain_memory is None:
            try:
                from langchain.memory import ConversationSummaryBufferMemory

                self._langchain_memory = ConversationSummaryBufferMemory(
                    max_token_limit=2000, return_messages=True
                )
                logger.info("LangChain memory initialized")
            except ImportError:
                logger.warning("LangChain not available")
                self._langchain_memory = None
        return self._langchain_memory

    @property
    def storage_client(self):
        """Lazy load Google Cloud Storage"""
        if self._storage_client is None:
            try:
                from google.cloud import storage

                self._storage_client = storage.Client()
                self.bucket = self._storage_client.bucket(
                    os.environ.get("GCS_BUCKET", "jarvis-storage")
                )
                logger.info("Google Cloud Storage initialized")
            except Exception as e:
                logger.warning(f"GCS not available: {e}")
                self._storage_client = SimpleStorage()
        return self._storage_client

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "initialize":
                return {"result": self.server_info}

            elif method == "tools/list":
                return {"result": {"tools": list(self.tools.values())}}

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})
                result = await self.call_tool(tool_name, tool_params)
                return {"result": result}

            elif method == "prompts/list":
                return {"result": {"prompts": list(self.prompts.values())}}

            elif method == "prompts/get":
                prompt_name = params.get("name")
                prompt_args = params.get("arguments", {})
                result = self.get_prompt(prompt_name, prompt_args)
                return {"result": result}

            else:
                return {
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {"error": {"code": -32603, "message": str(e)}}

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        if tool_name == "store_memory":
            return await self.store_memory(params)
        elif tool_name == "retrieve_memory":
            return await self.retrieve_memory(params)
        elif tool_name == "store_file":
            return await self.store_file(params)
        elif tool_name == "retrieve_file":
            return await self.retrieve_file(params)
        elif tool_name == "list_files":
            return await self.list_files(params)
        elif tool_name == "update_knowledge":
            return await self.update_knowledge(params)
        elif tool_name == "search_knowledge":
            return await self.search_knowledge(params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def store_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store in long-term memory"""
        content = params["content"]
        category = params.get("category", "general")
        metadata = params.get("metadata", {})

        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["category"] = category

        # Store in mem0
        memory_id = self.memory.add(content, metadata=metadata)

        # Also update LangChain memory if it's a conversation
        if category == "conversation" and self.langchain_memory:
            self.langchain_memory.save_context(
                {"input": content}, {"output": "Stored in memory"}
            )

        return {
            "status": "success",
            "memory_id": memory_id,
            "category": category,
            "timestamp": metadata["timestamp"],
        }

    async def retrieve_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve from memory"""
        query = params["query"]
        category = params.get("category")
        limit = params.get("limit", 10)

        # Search mem0
        results = self.memory.search(query, limit=limit)

        # Filter by category if specified
        if category:
            results = [
                r for r in results if r.get("metadata", {}).get("category") == category
            ]

        return {
            "status": "success",
            "memories": results,
            "count": len(results),
            "query": query,
        }

    async def store_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Store file in cloud storage"""
        file_path = params["file_path"]
        storage_path = params.get("storage_path", Path(file_path).name)
        metadata = params.get("metadata", {})

        if hasattr(self.storage_client, "bucket"):
            # Real GCS
            blob = self.bucket.blob(storage_path)
            blob.metadata = metadata
            blob.upload_from_filename(file_path)

            return {
                "status": "success",
                "storage_path": storage_path,
                "size": blob.size,
                "url": blob.public_url,
            }
        else:
            # Simple storage fallback
            return self.storage_client.store_file(file_path, storage_path, metadata)

    async def retrieve_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve file from storage"""
        storage_path = params["storage_path"]
        local_path = params.get("local_path", f"./downloads/{Path(storage_path).name}")

        # Ensure download directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.storage_client, "bucket"):
            # Real GCS
            blob = self.bucket.blob(storage_path)
            blob.download_to_filename(local_path)

            return {"status": "success", "local_path": local_path, "size": blob.size}
        else:
            # Simple storage fallback
            return self.storage_client.retrieve_file(storage_path, local_path)

    async def list_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List files in storage"""
        prefix = params.get("prefix", "")
        limit = params.get("limit", 100)

        if hasattr(self.storage_client, "bucket"):
            # Real GCS
            blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=limit))
            files = [
                {
                    "name": blob.name,
                    "size": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                }
                for blob in blobs
            ]
        else:
            # Simple storage fallback
            files = self.storage_client.list_files(prefix, limit)

        return {
            "status": "success",
            "files": files,
            "count": len(files),
            "prefix": prefix,
        }

    async def update_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge base"""
        topic = params["topic"]
        content = params["content"]
        source = params.get("source", "direct_input")

        # Store in knowledge base
        if topic not in self._knowledge_base:
            self._knowledge_base[topic] = []

        entry = {
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        self._knowledge_base[topic].append(entry)

        # Also store in memory for searchability
        await self.store_memory(
            {
                "content": f"Knowledge about {topic}: {content}",
                "category": "knowledge",
                "metadata": {"topic": topic, "source": source},
            }
        )

        return {
            "status": "success",
            "topic": topic,
            "entries": len(self._knowledge_base[topic]),
        }

    async def search_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search knowledge base"""
        query = params["query"]
        topic = params.get("topic")

        results = []

        # Search in knowledge base
        for kb_topic, entries in self._knowledge_base.items():
            if topic and kb_topic != topic:
                continue

            for entry in entries:
                if query.lower() in entry["content"].lower():
                    results.append(
                        {
                            "topic": kb_topic,
                            "content": entry["content"],
                            "source": entry["source"],
                            "timestamp": entry["timestamp"],
                            "relevance": 1.0,  # Simple keyword match
                        }
                    )

        # Also search in memory
        memory_results = await self.retrieve_memory(
            {"query": query, "category": "knowledge", "limit": 20}
        )

        # Merge results
        for mem in memory_results.get("memories", []):
            results.append(
                {
                    "topic": mem.get("metadata", {}).get("topic", "general"),
                    "content": mem.get("text", ""),
                    "source": "memory",
                    "relevance": mem.get("score", 0.5),
                }
            )

        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)

        return {
            "status": "success",
            "results": results[:20],  # Limit to top 20
            "count": len(results),
            "query": query,
        }

    def get_prompt(self, prompt_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a formatted prompt"""
        if prompt_name == "remember_conversation":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please remember this conversation summary for future reference: {args.get('conversation_summary', '')}",
                    }
                ]
            }
        elif prompt_name == "recall_context":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"What do you remember about: {args.get('current_topic', '')}?",
                    }
                ]
            }
        elif prompt_name == "save_learning":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Save this learning about {args.get('topic', 'general')}: {args.get('learning_content', '')}",
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown prompt: {prompt_name}")


class SimpleMemory:
    """Fallback memory implementation"""

    def __init__(self):
        self.memories = []

    def add(self, content: str, metadata: Dict = None) -> str:
        memory_id = f"mem_{len(self.memories)}"
        self.memories.append(
            {"id": memory_id, "text": content, "metadata": metadata or {}}
        )
        return memory_id

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        results = []
        for mem in self.memories:
            if query.lower() in mem["text"].lower():
                results.append(mem)
        return results[:limit]


class SimpleStorage:
    """Fallback storage implementation"""

    def __init__(self):
        self.storage_dir = Path("./storage/files")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def store_file(self, file_path: str, storage_path: str, metadata: Dict) -> Dict:
        dest = self.storage_dir / storage_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy2(file_path, dest)

        # Store metadata
        meta_file = dest.with_suffix(dest.suffix + ".meta")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        return {
            "status": "success",
            "storage_path": storage_path,
            "size": dest.stat().st_size,
        }

    def retrieve_file(self, storage_path: str, local_path: str) -> Dict:
        src = self.storage_dir / storage_path
        if not src.exists():
            raise FileNotFoundError(f"File not found: {storage_path}")

        import shutil

        shutil.copy2(src, local_path)

        return {
            "status": "success",
            "local_path": local_path,
            "size": src.stat().st_size,
        }

    def list_files(self, prefix: str, limit: int) -> List[Dict]:
        files = []
        for path in self.storage_dir.rglob(f"{prefix}*"):
            if path.is_file() and not path.suffix.endswith(".meta"):
                files.append(
                    {
                        "name": str(path.relative_to(self.storage_dir)),
                        "size": path.stat().st_size,
                        "updated": datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).isoformat(),
                    }
                )
        return files[:limit]


async def main():
    """Main entry point for MCP server"""
    server = JARVISMemoryStorageMCP()

    logger.info("JARVIS Memory & Storage MCP Server starting...")

    # MCP servers communicate via stdio
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer = sys.stdout

    while True:
        try:
            # Read request
            line = await reader.readline()
            if not line:
                break

            request = json.loads(line.decode())
            logger.debug(f"Request: {request}")

            # Handle request
            response = await server.handle_request(request)

            # Write response
            writer.write(json.dumps(response).encode() + b"\n")
            writer.flush()

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            error_response = {"error": {"code": -32603, "message": str(e)}}
            writer.write(json.dumps(error_response).encode() + b"\n")
            writer.flush()


if __name__ == "__main__":
    asyncio.run(main())
