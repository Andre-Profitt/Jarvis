#!/usr/bin/env python3
"""
Local-only Claude Memory RAG Server
No OpenAI API key required - uses only local models
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import asyncio

# MCP imports
from mcp import Server
from mcp.types import Resource, Tool, TextContent
from mcp.server.stdio import stdio_server

# Local embeddings and storage
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from google.cloud import storage
import hashlib
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Memory:
    id: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class LocalMemoryRAG:
    """Memory system using only local models - no OpenAI required"""

    def __init__(self):
        # Paths
        self.base_path = os.path.expanduser("~/.claude_local_memory")
        os.makedirs(self.base_path, exist_ok=True)

        # Initialize local embedding model
        logger.info("Loading local embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB with local embeddings
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.base_path, "chroma_db")
        )

        # Create collection with local embedding function
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="claude_memories", embedding_function=embedding_fn
        )

        # Initialize GCS if available
        self.gcs_client = None
        self.gcs_bucket = None
        try:
            self.gcs_client = storage.Client()
            self.gcs_bucket = self.gcs_client.bucket("jarvis-memory-storage")
            logger.info("✅ Connected to Google Cloud Storage")
        except Exception as e:
            logger.warning(f"GCS not available: {e}")

        # Memory index
        self.memory_index = self._load_memory_index()

    def _load_memory_index(self) -> Dict[str, Memory]:
        """Load memory index from disk"""
        index_path = os.path.join(self.base_path, "memory_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                data = json.load(f)
                return {k: Memory(**v) for k, v in data.items()}
        return {}

    def _save_memory_index(self):
        """Save memory index to disk"""
        index_path = os.path.join(self.base_path, "memory_index.json")
        data = {}
        for k, v in self.memory_index.items():
            memory_dict = {
                "id": v.id,
                "content": v.content,
                "timestamp": v.timestamp.isoformat(),
                "metadata": v.metadata,
            }
            data[k] = memory_dict

        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def store_memory(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a memory using local embeddings"""
        # Generate ID
        memory_id = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Store in ChromaDB
        self.collection.add(
            documents=[content], ids=[memory_id], metadatas=[metadata or {}]
        )

        # Store in local index
        self.memory_index[memory_id] = memory
        self._save_memory_index()

        # Backup to GCS if available
        if self.gcs_bucket:
            try:
                blob = self.gcs_bucket.blob(f"memories/{memory_id}.json")
                blob.upload_from_string(
                    json.dumps(
                        {
                            "content": content,
                            "metadata": metadata,
                            "timestamp": memory.timestamp.isoformat(),
                        }
                    )
                )
            except Exception as e:
                logger.error(f"GCS backup failed: {e}")

        logger.info(f"✅ Stored memory {memory_id}")
        return memory_id

    def search_memories(self, query: str, n_results: int = 5) -> List[Memory]:
        """Search memories using local embeddings"""
        results = self.collection.query(query_texts=[query], n_results=n_results)

        memories = []
        if results["ids"] and results["ids"][0]:
            for memory_id in results["ids"][0]:
                if memory_id in self.memory_index:
                    memories.append(self.memory_index[memory_id])

        return memories

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_memories": len(self.memory_index),
            "storage_path": self.base_path,
            "embedding_model": "all-MiniLM-L6-v2 (local)",
            "vector_store": "ChromaDB",
            "gcs_enabled": self.gcs_bucket is not None,
        }


# Create MCP server
server = Server("claude-memory-local")
memory_system = LocalMemoryRAG()


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available memory resources"""
    return [
        Resource(
            uri="memory://stats",
            name="Memory System Stats",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read memory resources"""
    if uri == "memory://stats":
        return json.dumps(memory_system.get_stats(), indent=2)
    return "{}"


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available memory tools"""
    return [
        Tool(
            name="store_memory",
            description="Store a memory or important information",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memories",
            description="Search through stored memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute memory tools"""
    if name == "store_memory":
        memory_id = memory_system.store_memory(
            arguments["content"], {"tags": arguments.get("tags", [])}
        )
        return [
            TextContent(
                type="text", text=f"Memory stored successfully with ID: {memory_id}"
            )
        ]

    elif name == "search_memories":
        memories = memory_system.search_memories(
            arguments["query"], arguments.get("n_results", 5)
        )

        if not memories:
            return [TextContent(type="text", text="No relevant memories found.")]

        results = []
        for memory in memories:
            results.append(
                f"[{memory.timestamp.strftime('%Y-%m-%d')}] {memory.content}"
            )

        return [TextContent(type="text", text="\n\n".join(results))]

    return [TextContent(type="text", text="Unknown tool")]


async def main():
    """Run the local memory server"""
    logger.info("🚀 Starting Local Memory RAG Server (No OpenAI required)")
    logger.info(f"📁 Storage: {memory_system.base_path}")
    logger.info("🧠 Using local sentence-transformers for embeddings")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
