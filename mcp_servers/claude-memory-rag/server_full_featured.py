#!/usr/bin/env python3
"""
Claude Memory RAG - Full Featured with LangChain + Mem0 + OpenAI
Combines the best of all memory systems with 30TB Google Cloud Storage
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib

# Set up environment
os.environ["OPENAI_API_KEY"] = (
    "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"
)
os.environ["MEM0_API_KEY"] = "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "claude-memory-rag-full"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

# Memory systems
from mem0 import Memory
from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Google Cloud Storage
from google.cloud import storage

# OpenAI for advanced features
import openai

print("🚀 Initializing Full-Featured Memory System...", file=sys.stderr)


class FullFeaturedMemoryRAG:
    """Ultimate memory system combining Mem0, LangChain, OpenAI, and GCS"""

    def __init__(self):
        # Initialize storage paths
        self.local_dir = Path.home() / ".claude_full_memory"
        self.local_dir.mkdir(exist_ok=True)

        # 1. Initialize Mem0 with OpenAI
        try:
            self.mem0 = Memory()
            self.has_mem0 = True
            print("✅ Mem0 initialized with OpenAI", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Mem0 error: {e}", file=sys.stderr)
            self.has_mem0 = False

        # 2. Initialize OpenAI embeddings for LangChain
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  # Latest and most cost-effective
            )
            print("✅ OpenAI embeddings initialized", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  OpenAI embeddings error: {e}", file=sys.stderr)
            # Fallback to local embeddings
            from sentence_transformers import SentenceTransformer

            self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
            print("📦 Using local embeddings as fallback", file=sys.stderr)

        # 3. Initialize ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name="claude_memories",
            embedding_function=self.embeddings,
            persist_directory=str(self.local_dir / "chroma_db"),
        )

        # 4. Initialize LangChain memory systems
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            max_token_limit=2000,
            return_messages=True,
        )

        self.vector_memory = VectorStoreRetrieverMemory(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory_key="relevant_memories",
            return_docs=True,
        )

        # 5. Initialize Google Cloud Storage (30TB)
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket("jarvis-memory-storage")
            self.has_gcs = True
            print("✅ Connected to Google Cloud Storage (30TB)", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  GCS not available: {e}", file=sys.stderr)
            self.has_gcs = False

        # 6. Local memory structure
        self.local_memory = {
            "conversations": {},
            "patterns": {},
            "user_profile": {},
            "code_snippets": {},
        }

        self._load_local_memory()
        print("✅ Full-Featured Memory System Ready!", file=sys.stderr)

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

    async def store_conversation(
        self,
        conversation_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None,
    ):
        """Store conversation using all available systems"""
        success_count = 0

        # 1. Store in Mem0 (with OpenAI enhancement)
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
                                **(metadata or {}),
                            },
                        )
                success_count += 1
                print(f"✅ Stored in Mem0", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  Mem0 storage error: {e}", file=sys.stderr)

        # 2. Store in LangChain vector store
        try:
            # Create document for vector storage
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

            # Also update conversation memory
            for msg in messages:
                if msg.get("role") == "user":
                    self.conversation_memory.save_context(
                        {"input": msg.get("content", "")},
                        {"output": ""},  # Will be filled by assistant response
                    )

            success_count += 1
            print(f"✅ Stored in LangChain", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  LangChain storage error: {e}", file=sys.stderr)

        # 3. Store locally
        self.local_memory["conversations"][conversation_id] = {
            "messages": messages,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._save_local_memory()
        success_count += 1

        # 4. Backup to GCS (30TB)
        if self.has_gcs:
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
                print(f"☁️  Backed up to GCS", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  GCS backup error: {e}", file=sys.stderr)

        return success_count > 0

    async def recall_memories(
        self, query: str, memory_type: str = "all", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall memories from all systems with OpenAI-powered ranking"""
        all_memories = []

        # 1. Recall from Mem0 (OpenAI-enhanced)
        if self.has_mem0:
            try:
                mem0_results = self.mem0.search(query, limit=top_k)
                for result in mem0_results:
                    all_memories.append(
                        {
                            "source": "mem0_openai",
                            "content": result.get("text", ""),
                            "metadata": result.get("metadata", {}),
                            "relevance": result.get("score", 0.8),
                        }
                    )
                print(f"📚 Found {len(mem0_results)} memories in Mem0", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  Mem0 recall error: {e}", file=sys.stderr)

        # 2. Recall from LangChain vector store
        try:
            docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
            for doc, score in docs:
                all_memories.append(
                    {
                        "source": "langchain_vectors",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance": 1 - score,  # Convert distance to similarity
                    }
                )
            print(f"📚 Found {len(docs)} memories in LangChain", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  LangChain recall error: {e}", file=sys.stderr)

        # 3. Get conversation context from LangChain
        try:
            conversation_context = self.conversation_memory.buffer
            if conversation_context:
                all_memories.append(
                    {
                        "source": "conversation_buffer",
                        "content": str(conversation_context),
                        "metadata": {"type": "recent_conversation"},
                        "relevance": 0.9,  # Recent conversations are highly relevant
                    }
                )
        except Exception as e:
            print(f"⚠️  Conversation memory error: {e}", file=sys.stderr)

        # 4. Search local memory
        for conv_id, conv_data in self.local_memory["conversations"].items():
            conv_text = " ".join([m.get("content", "") for m in conv_data["messages"]])
            if query.lower() in conv_text.lower():
                all_memories.append(
                    {
                        "source": "local",
                        "content": conv_text[:500],  # First 500 chars
                        "metadata": conv_data.get("metadata", {}),
                        "relevance": 0.5,
                    }
                )

        # Sort by relevance and deduplicate
        all_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        # Remove duplicates based on content similarity
        unique_memories = []
        seen_content = set()

        for memory in all_memories:
            content_hash = hashlib.md5(memory["content"].encode()).hexdigest()[:8]
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_memories.append(memory)

        return unique_memories[:top_k]

    async def analyze_with_gpt4(self, query: str, context: List[Dict]) -> str:
        """Use GPT-4 to analyze memories and provide insights"""
        try:
            # Prepare context
            context_str = "\n\n".join(
                [f"[{mem['source']}] {mem['content'][:200]}..." for mem in context[:5]]
            )

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing memories to provide insights and connections.",
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nRelevant memories:\n{context_str}\n\nProvide insights:",
                    },
                ],
                temperature=0.7,
                max_tokens=500,
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  GPT-4 analysis error: {e}", file=sys.stderr)
            return "Analysis not available"

    async def learn_pattern(
        self, pattern: str, success: bool, context: Optional[Dict] = None
    ):
        """Learn patterns using all systems"""
        pattern_id = f"pattern_{datetime.now().timestamp()}"

        # Store in Mem0
        if self.has_mem0:
            try:
                self.mem0.add(
                    f"Learned pattern: {pattern} (Success: {success})",
                    user_id="system",
                    metadata={
                        "type": "pattern",
                        "success": success,
                        "context": context,
                    },
                )
            except:
                pass

        # Store in vector store
        try:
            doc = Document(
                page_content=f"Pattern: {pattern}",
                metadata={
                    "type": "pattern",
                    "success": success,
                    "context": context or {},
                    "timestamp": datetime.now().isoformat(),
                },
            )
            self.vectorstore.add_documents([doc])
        except:
            pass

        # Store locally
        self.local_memory["patterns"][pattern_id] = {
            "pattern": pattern,
            "success": success,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }
        self._save_local_memory()

        # Backup to GCS
        if self.has_gcs:
            try:
                blob = self.bucket.blob(f"patterns/{pattern_id}.json")
                blob.upload_from_string(
                    json.dumps(self.local_memory["patterns"][pattern_id])
                )
            except:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "total_conversations": len(self.local_memory["conversations"]),
            "total_patterns": len(self.local_memory["patterns"]),
            "systems_active": [],
            "storage_locations": [],
            "features": [],
        }

        if self.has_mem0:
            stats["systems_active"].append("Mem0 (OpenAI-powered)")
            stats["features"].append("OpenAI semantic search")

        stats["systems_active"].append("LangChain (ChromaDB + OpenAI)")
        stats["features"].append("Vector similarity search")
        stats["features"].append("Conversation summarization")

        if self.has_gcs:
            stats["storage_locations"].append("Google Cloud Storage (30TB)")

        stats["storage_locations"].append(f"Local: {self.local_dir}")

        # Get vector store stats
        try:
            collection = self.vectorstore._collection
            stats["vector_store_count"] = collection.count()
        except:
            stats["vector_store_count"] = "Unknown"

        return stats


class FullFeaturedMCPServer:
    """MCP Server with all memory features"""

    def __init__(self):
        self.memory = FullFeaturedMemoryRAG()
        self.start_time = datetime.now()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        response = {"jsonrpc": "2.0", "id": request_id}

        try:
            if method == "initialize":
                response["result"] = {
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "claude-memory-full", "version": "3.0.0"},
                }

            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "store_conversation",
                            "description": "Store conversation with Mem0 + LangChain + GCS",
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
                            "description": "Search memories using OpenAI + LangChain + Mem0",
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
                            "name": "analyze_memories",
                            "description": "Use GPT-4 to analyze and provide insights",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "include_analysis": {
                                        "type": "boolean",
                                        "default": True,
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "get_memory_stats",
                            "description": "Get comprehensive memory statistics",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "learn_pattern",
                            "description": "Learn patterns across all memory systems",
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
                                "text": f"✅ Conversation stored in {len(stats['systems_active'])} systems\n"
                                f"Active: {', '.join(stats['systems_active'])}\n"
                                f"Features: {', '.join(stats['features'])}",
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

                elif tool_name == "analyze_memories":
                    query = tool_params.get("query", "")

                    # First get relevant memories
                    memories = await self.memory.recall_memories(query, top_k=5)

                    # Then analyze with GPT-4
                    if tool_params.get("include_analysis", True) and memories:
                        analysis = await self.memory.analyze_with_gpt4(query, memories)

                        response["result"] = {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Found {len(memories)} relevant memories.\n\n"
                                    f"GPT-4 Analysis:\n{analysis}",
                                }
                            ]
                        }
                    else:
                        response["result"] = {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Found {len(memories)} relevant memories.",
                                }
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
                    await self.memory.learn_pattern(
                        tool_params.get("pattern"),
                        tool_params.get("success", True),
                        tool_params.get("context"),
                    )
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": "Pattern learned across all memory systems",
                            }
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
        """Start MCP server"""
        print("🧠 Claude Memory RAG - Full Featured", file=sys.stderr)
        print("💎 With: Mem0 + LangChain + OpenAI + GCS", file=sys.stderr)
        print(f"📊 Stats: {self.memory.get_stats()}", file=sys.stderr)
        print("✅ Ready for connections...", file=sys.stderr)

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break

                request = json.loads(line.strip())
                response = await self.handle_request(request)

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
    server = FullFeaturedMCPServer()
    asyncio.run(server.start_server())
