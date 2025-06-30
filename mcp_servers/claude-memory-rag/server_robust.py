#!/usr/bin/env python3
"""
Claude Memory RAG - Robust Version with Fallback Support
Handles missing dependencies gracefully
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import logging
import traceback

# Set up logging
log_dir = Path.home() / ".claude_full_memory" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "mcp_server.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

# Set up environment (if not already set)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = (
        "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"
    )
if "MEM0_API_KEY" not in os.environ:
    os.environ["MEM0_API_KEY"] = "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC"

# Try to import dependencies with fallbacks
has_mem0 = False
has_langchain = False
has_gcs = False
has_openai = False

try:
    from mem0 import Memory

    has_mem0 = True
    logger.info("✅ Mem0 available")
except ImportError as e:
    logger.warning(f"⚠️  Mem0 not available: {e}")

try:
    from langchain.memory import (
        ConversationSummaryBufferMemory,
        VectorStoreRetrieverMemory,
    )
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document

    has_langchain = True
    logger.info("✅ LangChain available")
except ImportError as e:
    logger.warning(f"⚠️  LangChain not available: {e}")

try:
    from google.cloud import storage

    has_gcs = True
    logger.info("✅ Google Cloud Storage available")
except ImportError as e:
    logger.warning(f"⚠️  Google Cloud Storage not available: {e}")

try:
    import openai

    has_openai = True
    logger.info("✅ OpenAI available")
except ImportError as e:
    logger.warning(f"⚠️  OpenAI not available: {e}")

logger.info("🚀 Initializing Robust Memory System...")


class RobustMemoryRAG:
    """Memory system that works with whatever is available"""

    def __init__(self):
        # Initialize storage paths
        self.local_dir = Path.home() / ".claude_full_memory"
        self.local_dir.mkdir(exist_ok=True)

        # Initialize available systems
        self.mem0 = None
        self.vectorstore = None
        self.conversation_memory = None
        self.storage_client = None
        self.embeddings = None

        # 1. Initialize Mem0 if available
        if has_mem0:
            try:
                self.mem0 = Memory()
                logger.info("✅ Mem0 initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0: {e}")

        # 2. Initialize LangChain if available
        if has_langchain:
            try:
                # Try OpenAI embeddings first
                if has_openai:
                    self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    logger.info("✅ OpenAI embeddings initialized")
                else:
                    # Fallback to sentence transformers
                    try:
                        from sentence_transformers import SentenceTransformer

                        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
                        logger.info("📦 Using local embeddings")
                    except:
                        logger.warning("No embeddings available")

                # Initialize vector store if embeddings are available
                if self.embeddings:
                    self.vectorstore = Chroma(
                        collection_name="claude_memories",
                        embedding_function=self.embeddings,
                        persist_directory=str(self.local_dir / "chroma_db"),
                    )

                    # Initialize conversation memory
                    if has_openai:
                        self.conversation_memory = ConversationSummaryBufferMemory(
                            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                            max_token_limit=2000,
                            return_messages=True,
                        )
                        logger.info("✅ Conversation memory initialized")

            except Exception as e:
                logger.error(f"Failed to initialize LangChain: {e}")

        # 3. Initialize GCS if available
        if has_gcs:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket("jarvis-memory-storage")
                logger.info("✅ Connected to Google Cloud Storage")
            except Exception as e:
                logger.warning(f"Failed to connect to GCS: {e}")
                self.storage_client = None

        # 4. Local memory structure (always available)
        self.local_memory = {
            "conversations": {},
            "patterns": {},
            "user_profile": {},
            "code_snippets": {},
        }

        self._load_local_memory()
        logger.info("✅ Robust Memory System Ready!")

    def _load_local_memory(self):
        """Load local memory from disk"""
        memory_file = self.local_dir / "memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, "r") as f:
                    self.local_memory = json.load(f)
                logger.info(
                    f"Loaded {len(self.local_memory.get('conversations', {}))} conversations from disk"
                )
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")

    def _save_local_memory(self):
        """Save local memory to disk"""
        memory_file = self.local_dir / "memory.json"
        try:
            with open(memory_file, "w") as f:
                json.dump(self.local_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    async def store_conversation(
        self,
        conversation_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None,
    ):
        """Store conversation using all available systems"""
        success_count = 0

        # Always store locally
        self.local_memory["conversations"][conversation_id] = {
            "messages": messages,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._save_local_memory()
        success_count += 1
        logger.info(f"✅ Stored conversation {conversation_id} locally")

        # Store in Mem0 if available
        if self.mem0:
            try:
                for msg in messages:
                    if msg.get("role") == "user":
                        self.mem0.add(
                            msg.get("content", ""),
                            user_id=(
                                metadata.get("user_id", "default")
                                if metadata
                                else "default"
                            ),
                            metadata={
                                "conversation_id": conversation_id,
                                "timestamp": datetime.now().isoformat(),
                                **(metadata or {}),
                            },
                        )
                success_count += 1
                logger.info("✅ Stored in Mem0")
            except Exception as e:
                logger.error(f"Mem0 storage error: {e}")

        # Store in vector store if available
        if self.vectorstore:
            try:
                conversation_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in messages]
                )

                doc = Document(
                    page_content=conversation_text,
                    metadata={
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {}),
                    },
                )

                self.vectorstore.add_documents([doc])
                success_count += 1
                logger.info("✅ Stored in vector store")
            except Exception as e:
                logger.error(f"Vector store error: {e}")

        # Backup to GCS if available
        if self.storage_client and self.bucket:
            try:
                blob = self.bucket.blob(f"conversations/{conversation_id}.json")
                blob.upload_from_string(
                    json.dumps(
                        {
                            "messages": messages,
                            "metadata": metadata,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                )
                success_count += 1
                logger.info("☁️  Backed up to GCS")
            except Exception as e:
                logger.error(f"GCS backup error: {e}")

        return success_count > 0

    async def recall_memories(
        self, query: str, memory_type: str = "all", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall memories from all available systems"""
        all_memories = []

        # Search local memory (always available)
        try:
            for conv_id, conv_data in self.local_memory["conversations"].items():
                conv_text = " ".join(
                    [m.get("content", "") for m in conv_data["messages"]]
                )
                if query.lower() in conv_text.lower():
                    all_memories.append(
                        {
                            "source": "local",
                            "content": conv_text[:500],
                            "metadata": conv_data.get("metadata", {}),
                            "relevance": 0.5,
                        }
                    )
            logger.info(f"Found {len(all_memories)} memories locally")
        except Exception as e:
            logger.error(f"Local search error: {e}")

        # Search Mem0 if available
        if self.mem0:
            try:
                mem0_results = self.mem0.search(query, limit=top_k)
                for result in mem0_results:
                    all_memories.append(
                        {
                            "source": "mem0",
                            "content": result.get("text", ""),
                            "metadata": result.get("metadata", {}),
                            "relevance": result.get("score", 0.8),
                        }
                    )
                logger.info(f"Found {len(mem0_results)} memories in Mem0")
            except Exception as e:
                logger.error(f"Mem0 recall error: {e}")

        # Search vector store if available
        if self.vectorstore:
            try:
                docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
                for doc, score in docs:
                    all_memories.append(
                        {
                            "source": "vectorstore",
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance": 1 - score,
                        }
                    )
                logger.info(f"Found {len(docs)} memories in vector store")
            except Exception as e:
                logger.error(f"Vector store recall error: {e}")

        # Sort by relevance and deduplicate
        all_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        # Remove duplicates
        unique_memories = []
        seen_content = set()

        for memory in all_memories:
            content_hash = hashlib.md5(memory["content"].encode()).hexdigest()[:8]
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_memories.append(memory)

        return unique_memories[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "total_conversations": len(self.local_memory["conversations"]),
            "total_patterns": len(self.local_memory["patterns"]),
            "systems_active": ["Local Storage"],
            "storage_locations": [str(self.local_dir)],
            "features": ["Local JSON storage"],
        }

        if self.mem0:
            stats["systems_active"].append("Mem0")
            stats["features"].append("Mem0 semantic search")

        if self.vectorstore:
            stats["systems_active"].append("ChromaDB")
            stats["features"].append("Vector similarity search")

        if self.conversation_memory:
            stats["features"].append("Conversation summarization")

        if self.storage_client:
            stats["systems_active"].append("Google Cloud Storage")
            stats["storage_locations"].append("GCS: jarvis-memory-storage")

        return stats


class RobustMCPServer:
    """MCP Server that works with available memory systems"""

    def __init__(self):
        self.memory = RobustMemoryRAG()
        self.start_time = datetime.now()
        logger.info("MCP Server initialized")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        logger.debug(f"Handling request: method={method}, id={request_id}")

        response = {"jsonrpc": "2.0", "id": request_id}

        try:
            if method == "initialize":
                response["result"] = {
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "claude-memory-robust", "version": "1.0.0"},
                }

            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "store_conversation",
                            "description": "Store conversation in available memory systems",
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
                            "description": "Search memories across available systems",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "memory_type": {"type": "string", "default": "all"},
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
                    ]
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                logger.info(f"Calling tool: {tool_name}")

                if tool_name == "store_conversation":
                    success = await self.memory.store_conversation(
                        tool_params.get(
                            "conversation_id", f"conv_{datetime.now().timestamp()}"
                        ),
                        tool_params.get("messages", []),
                        tool_params.get("metadata"),
                    )
                    stats = self.memory.get_stats()
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"✅ Conversation stored successfully\n"
                                f"Active systems: {', '.join(stats['systems_active'])}\n"
                                f"Total conversations: {stats['total_conversations']}",
                            }
                        ]
                    }

                elif tool_name == "recall_memories":
                    memories = await self.memory.recall_memories(
                        tool_params.get("query", ""),
                        tool_params.get("memory_type", "all"),
                        tool_params.get("top_k", 5),
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
                            memory_texts.append(
                                f"Memory {i+1} [{mem['source']}] (relevance: {mem['relevance']:.2f}):\n"
                                f"{mem['content'][:200]}..."
                            )

                        response["result"] = {
                            "content": [
                                {"type": "text", "text": "\n\n".join(memory_texts)}
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
            logger.error(f"Error handling request: {e}")
            logger.error(traceback.format_exc())
            response["error"] = {"code": -32603, "message": f"Internal error: {str(e)}"}

        return response

    async def start_server(self):
        """Start MCP server"""
        logger.info("🧠 Claude Memory RAG - Robust Version")
        logger.info(f"📊 Stats: {self.memory.get_stats()}")
        logger.info("✅ Ready for connections...")

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    logger.info("EOF received, shutting down")
                    break

                line = line.strip()
                if not line:
                    continue

                logger.debug(f"Received: {line[:100]}...")

                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                        "id": None,
                    }
                    sys.stdout.write(json.dumps(error_response) + "\n")
                    sys.stdout.flush()
                    continue

                response = await self.handle_request(request)

                response_str = json.dumps(response)
                logger.debug(f"Sending: {response_str[:100]}...")
                sys.stdout.write(response_str + "\n")
                sys.stdout.flush()

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                logger.error(traceback.format_exc())


if __name__ == "__main__":
    try:
        server = RobustMCPServer()
        asyncio.run(server.start_server())
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
