#!/usr/bin/env python3
"""
Simplified RAG Server - Works with minimal dependencies and proper MCP protocol
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

# Google Cloud Storage
try:
    from google.cloud import storage

    HAS_GCS = True
except:
    HAS_GCS = False


# Simple embeddings
class SimpleEmbeddings:
    def __init__(self, dim=384):
        self.dim = dim

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        # Simple but consistent embedding
        words = text.lower().split()[:50]
        vec = np.zeros(self.dim)
        for i, word in enumerate(words):
            idx = hash(word) % self.dim
            vec[idx] += 1.0 / (i + 1)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class SimpleVectorStore:
    """Simple in-memory vector store"""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.embedder = SimpleEmbeddings()

    def add_texts(self, texts, metadatas=None):
        """Add texts to the store"""
        for i, text in enumerate(texts):
            self.documents.append(text)
            self.embeddings.append(self.embedder.embed_query(text))
            self.metadatas.append(metadatas[i] if metadatas else {})

    def similarity_search(self, query, k=5):
        """Search for similar documents"""
        if not self.documents:
            return []

        query_embedding = np.array(self.embedder.embed_query(query))

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, np.array(emb))
            similarities.append((i, similarity))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in similarities[:k]:
            results.append(
                {
                    "content": self.documents[i],
                    "metadata": self.metadatas[i],
                    "score": score,
                }
            )

        return results


class SimplifiedRAG:
    """Simplified RAG that works reliably"""

    def __init__(self):
        # Local storage
        self.local_dir = Path.home() / ".claude_simple_memory"
        self.local_dir.mkdir(exist_ok=True)

        # Vector store
        self.vector_store = SimpleVectorStore()

        # Memory
        self.conversations = {}
        self.patterns = {}

        # GCS
        if HAS_GCS:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket("jarvis-memory-storage")
                self.has_gcs = True
                print("✅ Connected to GCS", file=sys.stderr)
            except:
                self.has_gcs = False
                print("⚠️  GCS not available", file=sys.stderr)
        else:
            self.has_gcs = False

        self._load_memory()

    def _load_memory(self):
        """Load memory from disk"""
        memory_file = self.local_dir / "memory.json"
        if memory_file.exists():
            with open(memory_file, "r") as f:
                data = json.load(f)
                self.conversations = data.get("conversations", {})
                self.patterns = data.get("patterns", {})

                # Rebuild vector store
                for conv_id, conv in self.conversations.items():
                    text = " ".join(
                        [m.get("content", "") for m in conv.get("messages", [])]
                    )
                    self.vector_store.add_texts([text], [{"id": conv_id}])

    def _save_memory(self):
        """Save memory to disk"""
        memory_file = self.local_dir / "memory.json"
        with open(memory_file, "w") as f:
            json.dump(
                {"conversations": self.conversations, "patterns": self.patterns},
                f,
                indent=2,
            )

    async def store_conversation(self, conv_id, messages, metadata=None):
        """Store a conversation"""
        # Store locally
        self.conversations[conv_id] = {
            "messages": messages,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Add to vector store
        text = " ".join([m.get("content", "") for m in messages])
        self.vector_store.add_texts([text], [{"id": conv_id, **(metadata or {})}])

        # Save
        self._save_memory()

        # Backup to GCS if available
        if self.has_gcs:
            try:
                blob = self.bucket.blob(f"conversations/{conv_id}.json")
                blob.upload_from_string(json.dumps(self.conversations[conv_id]))
            except:
                pass

        return True

    async def recall_memories(self, query, top_k=5):
        """Recall relevant memories"""
        # Search vector store
        results = self.vector_store.similarity_search(query, k=top_k)

        # Enhance with full conversation data
        memories = []
        for result in results:
            conv_id = result["metadata"].get("id")
            if conv_id in self.conversations:
                memories.append(
                    {
                        "conversation": self.conversations[conv_id],
                        "relevance": result["score"],
                        "metadata": result["metadata"],
                    }
                )

        return memories

    async def learn_pattern(self, pattern, success, context=None):
        """Learn a pattern"""
        pattern_id = f"pattern_{datetime.now().timestamp()}"
        self.patterns[pattern_id] = {
            "pattern": pattern,
            "success": success,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._save_memory()

    def get_stats(self):
        """Get memory statistics"""
        return {
            "total_conversations": len(self.conversations),
            "total_patterns": len(self.patterns),
            "vector_store_size": len(self.vector_store.documents),
            "storage": "GCS + Local" if self.has_gcs else "Local only",
        }


class SimpleRAGServer:
    """MCP server with proper protocol implementation"""

    def __init__(self):
        self.rag = SimplifiedRAG()
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
                    "serverInfo": {"name": "claude-memory-rag", "version": "1.0.0"},
                }

            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "store_conversation",
                            "description": "Store a conversation in memory with vector embeddings",
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
                            "name": "recall_memories",
                            "description": "Search and recall relevant memories using vector similarity",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer", "default": 5},
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
                    success = await self.rag.store_conversation(
                        tool_params.get(
                            "conversation_id", f"conv_{datetime.now().timestamp()}"
                        ),
                        tool_params.get("messages", []),
                        tool_params.get("metadata"),
                    )
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Conversation stored successfully. Storage: {self.rag.get_stats()['storage']}",
                            }
                        ]
                    }

                elif tool_name == "recall_memories":
                    memories = await self.rag.recall_memories(
                        tool_params.get("query", ""), tool_params.get("top_k", 5)
                    )

                    if not memories:
                        response["result"] = {
                            "content": [
                                {"type": "text", "text": "No relevant memories found."}
                            ]
                        }
                    else:
                        memory_texts = []
                        for i, mem in enumerate(memories):
                            conv = mem["conversation"]
                            memory_texts.append(
                                f"Memory {i+1} (relevance: {mem['relevance']:.2f}):\n"
                                f"  Timestamp: {conv.get('timestamp', 'Unknown')}\n"
                                f"  Messages: {len(conv.get('messages', []))}\n"
                                f"  Preview: {conv.get('messages', [{}])[0].get('content', 'No content')[:100]}..."
                            )

                        response["result"] = {
                            "content": [
                                {"type": "text", "text": "\n\n".join(memory_texts)}
                            ]
                        }

                elif tool_name == "get_memory_stats":
                    stats = self.rag.get_stats()
                    stats["uptime"] = str(datetime.now() - self.start_time)
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": json.dumps(stats, indent=2)}
                        ]
                    }

                elif tool_name == "learn_pattern":
                    await self.rag.learn_pattern(
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
        """Start MCP server with proper JSON-RPC communication"""
        print("🧠 Claude Memory RAG Server (Simplified)", file=sys.stderr)
        print(f"📊 Stats: {self.rag.get_stats()}", file=sys.stderr)
        print("✅ Ready for connections...", file=sys.stderr)

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
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(Path.home() / ".gcs/jarvis-credentials.json"),
    )

    server = SimpleRAGServer()
    asyncio.run(server.start_server())
