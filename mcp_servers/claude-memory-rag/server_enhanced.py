#!/usr/bin/env python3
"""
Enhanced Claude Memory RAG with Mem0 and LangChain
Combines multiple memory systems for ultimate capability
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

# Set environment variables
os.environ["MEM0_API_KEY"] = "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "claude-memory-rag"

# Google Cloud
from google.cloud import storage

# Memory systems
from mem0 import Memory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document


# Fallback embeddings
class LocalEmbeddings:
    """Local embeddings without OpenAI"""

    def __init__(self):
        self.dimension = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Simple but functional embeddings"""
        embeddings = []
        for text in texts:
            # Create deterministic embedding
            words = text.lower().split()[:50]
            embedding = np.zeros(self.dimension)

            for i, word in enumerate(words):
                # Use hash for consistency
                hash_val = hash(word) % self.dimension
                embedding[hash_val] = 1.0 / (i + 1)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding.tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]


class EnhancedClaudeMemoryRAG:
    """Enhanced RAG combining Mem0, LangChain, and custom systems"""

    def __init__(self, gcs_bucket: str = "jarvis-memory-storage"):
        # Google Cloud Storage
        self.gcs_bucket = gcs_bucket
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(gcs_bucket)
            print("✅ Connected to Google Cloud Storage")
        except Exception as e:
            print(f"⚠️  GCS not available: {e}")
            self.bucket = None

        # Local storage
        self.local_dir = Path.home() / ".claude_enhanced_memory"
        self.local_dir.mkdir(exist_ok=True)

        # Initialize Mem0
        try:
            self.mem0 = Memory()
            self.has_mem0 = True
            print("✅ Mem0 initialized with API key")
        except Exception as e:
            print(f"⚠️  Mem0 not available: {e}")
            self.has_mem0 = False

        # Initialize LangChain memory
        try:
            # Try OpenAI embeddings first, fallback to local
            try:
                embeddings = OpenAIEmbeddings()
            except:
                print("Using local embeddings")
                embeddings = LocalEmbeddings()

            # Vector store
            self.vectorstore = FAISS.from_texts(["Initial memory"], embeddings)

            # Conversation memory
            self.conversation_memory = ConversationSummaryBufferMemory(
                max_token_limit=2000, return_messages=True
            )

            self.has_langchain = True
            print("✅ LangChain memory initialized")
        except Exception as e:
            print(f"⚠️  LangChain not available: {e}")
            self.has_langchain = False

        # Fallback memory
        self.local_memory = {
            "conversations": {},
            "code_understanding": {},
            "patterns": {},
        }

        self._load_local_memory()

    def _load_local_memory(self):
        """Load local memory from disk"""
        memory_file = self.local_dir / "memory.json"
        if memory_file.exists():
            with open(memory_file, "r") as f:
                self.local_memory = json.load(f)

    def _save_local_memory(self):
        """Save local memory to disk"""
        memory_file = self.local_dir / "memory.json"
        with open(memory_file, "w") as f:
            json.dump(self.local_memory, f, indent=2)

    async def store_conversation_memory(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store conversation using all available systems"""
        try:
            success_count = 0

            # 1. Store in Mem0
            if self.has_mem0:
                try:
                    for msg in messages:
                        if msg.get("role") == "user":
                            self.mem0.add(
                                msg.get("content", ""),
                                user_id=metadata.get("user_id", "default"),
                                metadata={
                                    "conversation_id": conversation_id,
                                    "timestamp": datetime.now().isoformat(),
                                    **metadata,
                                },
                            )
                    success_count += 1
                except Exception as e:
                    print(f"Mem0 storage error: {e}")

            # 2. Store in LangChain
            if self.has_langchain:
                try:
                    # Add to conversation memory
                    for msg in messages:
                        if msg.get("role") == "user":
                            self.conversation_memory.save_context(
                                {"input": msg.get("content", "")}, {"output": ""}
                            )

                    # Add to vector store
                    conversation_text = " ".join(
                        [m.get("content", "") for m in messages]
                    )
                    self.vectorstore.add_texts(
                        [conversation_text],
                        metadatas=[
                            {
                                "conversation_id": conversation_id,
                                "timestamp": datetime.now().isoformat(),
                                **metadata,
                            }
                        ],
                    )
                    success_count += 1
                except Exception as e:
                    print(f"LangChain storage error: {e}")

            # 3. Store locally (always)
            self.local_memory["conversations"][conversation_id] = {
                "messages": messages,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
            }
            self._save_local_memory()
            success_count += 1

            # 4. Backup to GCS
            if self.bucket:
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
                except Exception as e:
                    print(f"GCS backup error: {e}")

            print(f"✅ Stored in {success_count} systems")
            return success_count > 0

        except Exception as e:
            print(f"Storage error: {e}")
            return False

    async def recall_relevant_memories(
        self, query: str, memory_type: str = "all", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall memories from all systems"""
        all_memories = []

        # 1. Recall from Mem0
        if self.has_mem0:
            try:
                mem0_results = self.mem0.search(query, limit=top_k)
                for result in mem0_results:
                    all_memories.append(
                        {
                            "source": "mem0",
                            "content": result,
                            "relevance": result.get("score", 0.5),
                        }
                    )
            except Exception as e:
                print(f"Mem0 recall error: {e}")

        # 2. Recall from LangChain vector store
        if self.has_langchain:
            try:
                docs = self.vectorstore.similarity_search(query, k=top_k)
                for doc in docs:
                    all_memories.append(
                        {
                            "source": "langchain",
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance": 0.7,  # Similarity score
                        }
                    )
            except Exception as e:
                print(f"LangChain recall error: {e}")

        # 3. Search local memory
        for conv_id, conv_data in self.local_memory["conversations"].items():
            # Simple keyword search
            conv_text = " ".join([m.get("content", "") for m in conv_data["messages"]])
            if query.lower() in conv_text.lower():
                all_memories.append(
                    {"source": "local", "content": conv_data, "relevance": 0.5}
                )

        # Sort by relevance and return top results
        all_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return all_memories[:top_k]

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics from all memory systems"""
        stats = {"total_memories": 0, "systems_active": [], "storage_locations": []}

        if self.has_mem0:
            stats["systems_active"].append("Mem0 (with API)")
            # Mem0 doesn't expose count easily

        if self.has_langchain:
            stats["systems_active"].append("LangChain")
            # Get vector store size

        stats["systems_active"].append("Local Storage")
        stats["total_memories"] += len(self.local_memory["conversations"])

        if self.bucket:
            stats["storage_locations"].append("Google Cloud Storage (30TB)")

        stats["storage_locations"].append(f"Local: {self.local_dir}")

        return stats

    async def learn_pattern(self, pattern: str, success: bool, context: Dict = None):
        """Learn patterns using Mem0's learning capabilities"""
        if self.has_mem0:
            self.mem0.add(
                f"Pattern: {pattern} - Success: {success}",
                user_id="system",
                metadata={
                    "type": "learned_pattern",
                    "success": success,
                    "context": context,
                },
            )

        # Also store locally
        pattern_id = f"pattern_{datetime.now().timestamp()}"
        self.local_memory["patterns"][pattern_id] = {
            "pattern": pattern,
            "success": success,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_local_memory()


# MCP Server implementation
class EnhancedMCPServer:
    """MCP Server with enhanced memory capabilities"""

    def __init__(self):
        self.memory = EnhancedClaudeMemoryRAG()
        self.start_time = datetime.now()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "store_conversation":
                success = await self.memory.store_conversation_memory(
                    params.get("conversation_id"),
                    params.get("messages", []),
                    params.get("metadata"),
                )
                return {"result": {"success": success}}

            elif method == "recall_memories":
                memories = await self.memory.recall_relevant_memories(
                    params.get("query", ""),
                    params.get("memory_type", "all"),
                    params.get("top_k", 5),
                )
                return {"result": {"memories": memories}}

            elif method == "get_memory_stats":
                stats = await self.memory.get_memory_stats()
                stats["uptime"] = str(datetime.now() - self.start_time)
                return {"result": stats}

            elif method == "learn_pattern":
                await self.memory.learn_pattern(
                    params.get("pattern"),
                    params.get("success", True),
                    params.get("context"),
                )
                return {"result": {"success": True}}

            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            return {"error": str(e)}

    async def start_server(self):
        """Start the MCP server"""
        print("🧠 Enhanced Claude Memory RAG Server")
        print(
            "📊 Active systems:",
            ", ".join((await self.memory.get_memory_stats())["systems_active"]),
        )
        print(
            "💾 Storage:",
            ", ".join((await self.memory.get_memory_stats())["storage_locations"]),
        )
        print("=" * 60)
        print("Ready for connections...")

        while True:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, input)
                request = json.loads(line)

                # Handle request
                response = await self.handle_request(request)

                # Send response
                print(json.dumps(response))

            except Exception as e:
                print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    # Set up environment
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
            Path.home() / ".gcs/jarvis-credentials.json"
        )

    # Start server
    server = EnhancedMCPServer()
    asyncio.run(server.start_server())
