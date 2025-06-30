#!/usr/bin/env python3
"""
Claude Memory RAG MCP Server
Gives Claude persistent memory using RAG with 30TB Google Cloud Storage
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
from pathlib import Path

# Vector DB and ML imports
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import pickle


class ClaudeMemoryRAG:
    """RAG system for Claude's persistent memory"""

    def __init__(self, gcs_bucket: str = "jarvis-memory-storage"):
        # Google Cloud Storage setup
        self.gcs_bucket = gcs_bucket
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcs_bucket)

        # Local cache directory
        self.cache_dir = Path.home() / ".claude_memory_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Vector database setup (ChromaDB with GCS persistence)
        self.chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.cache_dir / "chroma"),
                anonymized_telemetry=False,
            )
        )

        # Collections for different memory types
        self.collections = {
            "conversations": self._get_or_create_collection("claude_conversations"),
            "code_understanding": self._get_or_create_collection("code_understanding"),
            "project_knowledge": self._get_or_create_collection("project_knowledge"),
            "learned_patterns": self._get_or_create_collection("learned_patterns"),
        }

        # Embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Memory metadata
        self.memory_metadata = {
            "total_memories": 0,
            "last_sync": None,
            "memory_types": {},
        }

        # Load metadata from GCS
        self._load_metadata()

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )

    def _load_metadata(self):
        """Load memory metadata from GCS"""
        try:
            blob = self.bucket.blob("claude_memory/metadata.json")
            if blob.exists():
                self.memory_metadata = json.loads(blob.download_as_text())
        except Exception as e:
            print(f"Creating new memory metadata: {e}")

    def _save_metadata(self):
        """Save memory metadata to GCS"""
        blob = self.bucket.blob("claude_memory/metadata.json")
        blob.upload_from_string(json.dumps(self.memory_metadata, indent=2))

    async def store_conversation_memory(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store a conversation in memory"""
        try:
            # Create conversation summary
            summary = self._summarize_conversation(messages)

            # Generate embeddings
            embedding = self.embedder.encode(summary).tolist()

            # Store in ChromaDB
            self.collections["conversations"].add(
                embeddings=[embedding],
                documents=[json.dumps(messages)],
                metadatas=[
                    {
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "summary": summary,
                        "message_count": len(messages),
                        **(metadata or {}),
                    }
                ],
                ids=[conversation_id],
            )

            # Backup to GCS
            blob = self.bucket.blob(
                f"claude_memory/conversations/{conversation_id}.json"
            )
            blob.upload_from_string(
                json.dumps(
                    {
                        "messages": messages,
                        "summary": summary,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat(),
                    },
                    indent=2,
                )
            )

            # Update metadata
            self.memory_metadata["total_memories"] += 1
            self.memory_metadata["memory_types"]["conversations"] = (
                self.memory_metadata["memory_types"].get("conversations", 0) + 1
            )
            self._save_metadata()

            return True

        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False

    async def store_code_understanding(
        self,
        file_path: str,
        code_content: str,
        analysis: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store code understanding in memory"""
        try:
            # Create code summary
            summary = f"File: {file_path}\nAnalysis: {analysis[:200]}..."

            # Generate embeddings from code + analysis
            combined_text = f"{file_path}\n{code_content[:500]}\n{analysis}"
            embedding = self.embedder.encode(combined_text).tolist()

            # Create unique ID
            code_id = hashlib.md5(
                f"{file_path}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

            # Store in ChromaDB
            self.collections["code_understanding"].add(
                embeddings=[embedding],
                documents=[
                    json.dumps(
                        {
                            "file_path": file_path,
                            "code": code_content,
                            "analysis": analysis,
                        }
                    )
                ],
                metadatas=[
                    {
                        "file_path": file_path,
                        "timestamp": datetime.now().isoformat(),
                        "language": file_path.split(".")[-1],
                        **(metadata or {}),
                    }
                ],
                ids=[code_id],
            )

            # Backup to GCS
            blob = self.bucket.blob(f"claude_memory/code/{code_id}.json")
            blob.upload_from_string(
                json.dumps(
                    {
                        "file_path": file_path,
                        "code": code_content,
                        "analysis": analysis,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat(),
                    },
                    indent=2,
                )
            )

            return True

        except Exception as e:
            print(f"Error storing code understanding: {e}")
            return False

    async def recall_relevant_memories(
        self, query: str, memory_type: str = "all", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories based on query"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()

            memories = []

            # Search in specified collections
            if memory_type == "all":
                collections_to_search = self.collections.values()
            else:
                collections_to_search = [self.collections.get(memory_type)]

            for collection in collections_to_search:
                if collection:
                    results = collection.query(
                        query_embeddings=[query_embedding], n_results=top_k
                    )

                    # Format results
                    for i in range(len(results["ids"][0])):
                        memory = {
                            "id": results["ids"][0][i],
                            "content": json.loads(results["documents"][0][i]),
                            "metadata": results["metadatas"][0][i],
                            "distance": (
                                results["distances"][0][i]
                                if "distances" in results
                                else None
                            ),
                        }
                        memories.append(memory)

            # Sort by relevance (distance)
            memories.sort(key=lambda x: x.get("distance", float("inf")))

            return memories[:top_k]

        except Exception as e:
            print(f"Error recalling memories: {e}")
            return []

    async def learn_from_interaction(
        self,
        interaction_type: str,
        pattern: str,
        success: bool,
        context: Optional[Dict] = None,
    ) -> bool:
        """Learn patterns from interactions"""
        try:
            # Create pattern description
            pattern_text = f"{interaction_type}: {pattern} - Success: {success}"
            embedding = self.embedder.encode(pattern_text).tolist()

            # Generate ID
            pattern_id = hashlib.md5(
                f"{pattern}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

            # Store learned pattern
            self.collections["learned_patterns"].add(
                embeddings=[embedding],
                documents=[
                    json.dumps(
                        {
                            "interaction_type": interaction_type,
                            "pattern": pattern,
                            "success": success,
                            "context": context,
                        }
                    )
                ],
                metadatas=[
                    {
                        "timestamp": datetime.now().isoformat(),
                        "success": success,
                        "interaction_type": interaction_type,
                    }
                ],
                ids=[pattern_id],
            )

            return True

        except Exception as e:
            print(f"Error learning pattern: {e}")
            return False

    def _summarize_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Create a summary of conversation for embedding"""
        # Simple summarization - in production, use LLM for better summaries
        topics = []
        for msg in messages:
            if msg.get("role") == "user":
                # Extract key topics from user messages
                topics.append(msg.get("content", "")[:100])

        return f"Conversation topics: {'; '.join(topics[:5])}"

    async def sync_to_gcs(self):
        """Sync local vector DB to GCS"""
        print("📤 Syncing memory to Google Cloud Storage...")

        # Persist ChromaDB
        self.chroma_client.persist()

        # Upload ChromaDB files to GCS
        chroma_dir = self.cache_dir / "chroma"
        for file_path in chroma_dir.rglob("*"):
            if file_path.is_file():
                blob_path = f"claude_memory/chroma/{file_path.relative_to(chroma_dir)}"
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))

        # Update sync timestamp
        self.memory_metadata["last_sync"] = datetime.now().isoformat()
        self._save_metadata()

        print("✅ Memory synced to GCS!")

    async def load_from_gcs(self):
        """Load memory from GCS"""
        print("📥 Loading memory from Google Cloud Storage...")

        # Download ChromaDB files from GCS
        blobs = self.storage_client.list_blobs(
            self.gcs_bucket, prefix="claude_memory/chroma/"
        )

        for blob in blobs:
            local_path = (
                self.cache_dir
                / "chroma"
                / blob.name.replace("claude_memory/chroma/", "")
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))

        print("✅ Memory loaded from GCS!")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "total_memories": self.memory_metadata["total_memories"],
            "memory_types": self.memory_metadata["memory_types"],
            "last_sync": self.memory_metadata["last_sync"],
            "collections": {},
        }

        # Get stats for each collection
        for name, collection in self.collections.items():
            stats["collections"][name] = {"count": collection.count()}

        return stats


# MCP Server Protocol Handler
class MCPMemoryServer:
    """MCP Server for Claude Memory RAG"""

    def __init__(self):
        self.memory = ClaudeMemoryRAG()
        self.tools = {
            "store_conversation": self.handle_store_conversation,
            "recall_memories": self.handle_recall_memories,
            "store_code_understanding": self.handle_store_code,
            "learn_pattern": self.handle_learn_pattern,
            "get_memory_stats": self.handle_get_stats,
            "sync_memory": self.handle_sync,
        }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get("method")
        params = request.get("params", {})

        if method in self.tools:
            result = await self.tools[method](params)
            return {"result": result}
        else:
            return {"error": f"Unknown method: {method}"}

    async def handle_store_conversation(self, params: Dict) -> Dict:
        """Store conversation in memory"""
        success = await self.memory.store_conversation_memory(
            conversation_id=params.get("conversation_id"),
            messages=params.get("messages", []),
            metadata=params.get("metadata"),
        )
        return {"success": success}

    async def handle_recall_memories(self, params: Dict) -> Dict:
        """Recall relevant memories"""
        memories = await self.memory.recall_relevant_memories(
            query=params.get("query", ""),
            memory_type=params.get("memory_type", "all"),
            top_k=params.get("top_k", 5),
        )
        return {"memories": memories}

    async def handle_store_code(self, params: Dict) -> Dict:
        """Store code understanding"""
        success = await self.memory.store_code_understanding(
            file_path=params.get("file_path"),
            code_content=params.get("code_content"),
            analysis=params.get("analysis"),
            metadata=params.get("metadata"),
        )
        return {"success": success}

    async def handle_learn_pattern(self, params: Dict) -> Dict:
        """Learn from interaction pattern"""
        success = await self.memory.learn_from_interaction(
            interaction_type=params.get("interaction_type"),
            pattern=params.get("pattern"),
            success=params.get("success", True),
            context=params.get("context"),
        )
        return {"success": success}

    async def handle_get_stats(self, params: Dict) -> Dict:
        """Get memory statistics"""
        return self.memory.get_memory_stats()

    async def handle_sync(self, params: Dict) -> Dict:
        """Sync memory to GCS"""
        await self.memory.sync_to_gcs()
        return {"success": True}

    async def start_server(self):
        """Start the MCP server"""
        print("🧠 Claude Memory RAG MCP Server starting...")

        # Load existing memory from GCS
        await self.memory.load_from_gcs()

        # Server loop
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


if __name__ == "__main__":
    # Start MCP server
    server = MCPMemoryServer()
    asyncio.run(server.start_server())
