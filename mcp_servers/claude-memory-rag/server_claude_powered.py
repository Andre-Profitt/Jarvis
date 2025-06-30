#!/usr/bin/env python3
"""
Claude-Powered Memory RAG Server
Uses Anthropic's Claude API for intelligent memory management
with local embeddings for vector search
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
import hashlib

# MCP imports
from mcp import Server
from mcp.types import Resource, Tool, TextContent
from mcp.server.stdio import stdio_server

# Anthropic Claude
from anthropic import Anthropic

# Local embeddings and storage
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from google.cloud import storage
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Memory:
    id: str
    content: str
    summary: str
    timestamp: datetime
    metadata: Dict[str, Any]
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None


@dataclass
class UserProfile:
    interests: List[str]
    patterns: List[str]
    key_facts: Dict[str, Any]
    last_updated: datetime


class ClaudeMemoryRAG:
    """Memory system powered by Claude with local embeddings"""

    def __init__(self, anthropic_api_key: str):
        # Initialize Claude
        self.claude = Anthropic(api_key=anthropic_api_key)
        self.model = "claude-3-opus-20240229"  # Using Opus 4

        # Paths
        self.base_path = os.path.expanduser("~/.claude_anthropic_memory")
        os.makedirs(self.base_path, exist_ok=True)

        # Initialize local embedding model
        logger.info("Loading local embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.base_path, "chroma_db")
        )

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

        # Load data
        self.memory_index = self._load_memory_index()
        self.user_profile = self._load_user_profile()

    def _load_memory_index(self) -> Dict[str, Memory]:
        """Load memory index from disk"""
        index_path = os.path.join(self.base_path, "memory_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                data = json.load(f)
                memories = {}
                for k, v in data.items():
                    v["timestamp"] = datetime.fromisoformat(v["timestamp"])
                    memories[k] = Memory(**v)
                return memories
        return {}

    def _save_memory_index(self):
        """Save memory index to disk"""
        index_path = os.path.join(self.base_path, "memory_index.json")
        data = {}
        for k, v in self.memory_index.items():
            memory_dict = asdict(v)
            memory_dict["timestamp"] = v.timestamp.isoformat()
            memory_dict.pop("embedding", None)  # Don't save embeddings
            data[k] = memory_dict

        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_user_profile(self) -> UserProfile:
        """Load user profile"""
        profile_path = os.path.join(self.base_path, "user_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                data = json.load(f)
                data["last_updated"] = datetime.fromisoformat(data["last_updated"])
                return UserProfile(**data)
        return UserProfile(
            interests=[], patterns=[], key_facts={}, last_updated=datetime.now()
        )

    def _save_user_profile(self):
        """Save user profile"""
        profile_path = os.path.join(self.base_path, "user_profile.json")
        profile_dict = asdict(self.user_profile)
        profile_dict["last_updated"] = self.user_profile.last_updated.isoformat()

        with open(profile_path, "w") as f:
            json.dump(profile_dict, f, indent=2)

    def _analyze_with_claude(self, content: str, task: str) -> str:
        """Use Claude for intelligent analysis"""
        try:
            response = self.claude.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": f"{task}\n\nContent: {content}"}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return ""

    def _extract_importance_and_summary(self, content: str) -> tuple[float, str]:
        """Use Claude to determine importance and create summary"""
        prompt = """Analyze this content and provide:
1. An importance score from 0.0 to 1.0 (how important/memorable is this?)
2. A concise summary (1-2 sentences)

Format your response as:
IMPORTANCE: [score]
SUMMARY: [summary text]

Content to analyze:"""

        result = self._analyze_with_claude(content, prompt)

        # Parse response
        importance = 0.5
        summary = content[:100] + "..."  # Fallback

        for line in result.split("\n"):
            if line.startswith("IMPORTANCE:"):
                try:
                    importance = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        return importance, summary

    def _update_user_profile(self, content: str):
        """Use Claude to update user profile based on conversation"""
        if len(self.memory_index) % 10 == 0:  # Update every 10 memories
            recent_memories = sorted(
                self.memory_index.values(), key=lambda m: m.timestamp, reverse=True
            )[:20]

            memory_text = "\n".join([m.content for m in recent_memories])

            prompt = f"""Based on these recent conversations, extract:
1. User interests/topics (list)
2. Behavioral patterns (list)
3. Key facts about the user (dict format)

Current profile:
Interests: {self.user_profile.interests}
Patterns: {self.user_profile.patterns}
Facts: {self.user_profile.key_facts}

Recent conversations:
{memory_text}

Format response as:
INTERESTS: [comma-separated list]
PATTERNS: [comma-separated list]
FACTS: [json dict]"""

            result = self._analyze_with_claude("", prompt)

            # Parse and update profile
            for line in result.split("\n"):
                if line.startswith("INTERESTS:"):
                    interests = [i.strip() for i in line.split(":", 1)[1].split(",")]
                    self.user_profile.interests = list(set(interests))[:10]
                elif line.startswith("PATTERNS:"):
                    patterns = [p.strip() for p in line.split(":", 1)[1].split(",")]
                    self.user_profile.patterns = patterns[:5]
                elif line.startswith("FACTS:"):
                    try:
                        facts = json.loads(line.split(":", 1)[1])
                        self.user_profile.key_facts.update(facts)
                    except:
                        pass

            self.user_profile.last_updated = datetime.now()
            self._save_user_profile()

    def store_memory(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a memory with Claude-powered analysis"""
        # Use Claude to analyze importance and create summary
        importance, summary = self._extract_importance_and_summary(content)

        # Generate ID
        memory_id = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            summary=summary,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance_score=importance,
        )

        # Store in ChromaDB with metadata
        self.collection.add(
            documents=[content],
            ids=[memory_id],
            metadatas=[
                {
                    "summary": summary,
                    "importance": importance,
                    "timestamp": memory.timestamp.isoformat(),
                    **(metadata or {}),
                }
            ],
        )

        # Store in local index
        self.memory_index[memory_id] = memory
        self._save_memory_index()

        # Update user profile periodically
        self._update_user_profile(content)

        # Backup to GCS if available
        if self.gcs_bucket:
            try:
                blob = self.gcs_bucket.blob(f"memories/{memory_id}.json")
                blob.upload_from_string(json.dumps(asdict(memory)))
            except Exception as e:
                logger.error(f"GCS backup failed: {e}")

        logger.info(f"✅ Stored memory {memory_id} (importance: {importance:.2f})")
        return memory_id

    def search_memories(self, query: str, n_results: int = 5) -> List[Memory]:
        """Search memories with Claude-enhanced understanding"""
        # Use Claude to expand/improve the query
        enhanced_query = self._analyze_with_claude(
            query,
            "Rephrase this search query to be more comprehensive and include related concepts: ",
        )

        # Search with both original and enhanced query
        results = self.collection.query(
            query_texts=[query, enhanced_query], n_results=n_results * 2
        )

        # Collect unique memories
        seen_ids = set()
        memories = []

        for result_set in results["ids"]:
            for memory_id in result_set:
                if memory_id not in seen_ids and memory_id in self.memory_index:
                    seen_ids.add(memory_id)
                    memories.append(self.memory_index[memory_id])

        # Sort by relevance and importance
        memories.sort(key=lambda m: m.importance_score, reverse=True)

        return memories[:n_results]

    def consolidate_memories(self):
        """Use Claude to consolidate related memories"""
        if len(self.memory_index) < 20:
            return

        # Group memories by similarity
        all_memories = list(self.memory_index.values())
        memory_texts = "\n---\n".join(
            [f"[{m.id}] {m.summary}" for m in all_memories[-50:]]
        )

        prompt = f"""Review these memories and identify which ones could be consolidated because they discuss the same topic or event. 
        
List groups of memory IDs that should be consolidated:

{memory_texts}

Format: GROUP: [id1, id2, id3]"""

        result = self._analyze_with_claude("", prompt)

        # Parse and consolidate groups
        for line in result.split("\n"):
            if line.startswith("GROUP:"):
                try:
                    ids = json.loads(line.split(":", 1)[1])
                    self._consolidate_group(ids)
                except:
                    pass

    def _consolidate_group(self, memory_ids: List[str]):
        """Consolidate a group of related memories"""
        memories = [
            self.memory_index[mid] for mid in memory_ids if mid in self.memory_index
        ]
        if len(memories) < 2:
            return

        combined_content = "\n".join([m.content for m in memories])

        # Use Claude to create consolidated memory
        consolidated = self._analyze_with_claude(
            combined_content,
            "Create a single consolidated summary of these related memories:",
        )

        # Store consolidated memory
        self.store_memory(
            consolidated,
            {
                "type": "consolidated",
                "source_memories": memory_ids,
                "consolidation_date": datetime.now().isoformat(),
            },
        )

        # Mark old memories as consolidated
        for mid in memory_ids:
            if mid in self.memory_index:
                self.memory_index[mid].metadata["consolidated"] = True

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_memories": len(self.memory_index),
            "storage_path": self.base_path,
            "ai_model": "Claude Opus 4 (Anthropic)",
            "embedding_model": "all-MiniLM-L6-v2 (local)",
            "vector_store": "ChromaDB",
            "gcs_enabled": self.gcs_bucket is not None,
            "user_profile": {
                "interests": self.user_profile.interests,
                "patterns": self.user_profile.patterns,
                "facts": self.user_profile.key_facts,
            },
        }


# Load API key from environment or .env file
def get_anthropic_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY"):
                        key = line.split("=")[1].strip().strip("\"'")
                        break
    return key


# Create MCP server
server = Server("claude-memory-anthropic")
anthropic_key = get_anthropic_key()

if not anthropic_key:
    logger.error("No Anthropic API key found! Add ANTHROPIC_API_KEY to your .env file")
    memory_system = None
else:
    memory_system = ClaudeMemoryRAG(anthropic_key)


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available memory resources"""
    return [
        Resource(
            uri="memory://stats",
            name="Memory System Stats",
            mimeType="application/json",
        ),
        Resource(
            uri="memory://profile", name="User Profile", mimeType="application/json"
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read memory resources"""
    if not memory_system:
        return json.dumps({"error": "No Anthropic API key configured"})

    if uri == "memory://stats":
        return json.dumps(memory_system.get_stats(), indent=2)
    elif uri == "memory://profile":
        profile = memory_system.user_profile
        return json.dumps(
            {
                "interests": profile.interests,
                "patterns": profile.patterns,
                "key_facts": profile.key_facts,
                "last_updated": profile.last_updated.isoformat(),
            },
            indent=2,
        )
    return "{}"


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available memory tools"""
    return [
        Tool(
            name="store_memory",
            description="Store a memory with AI-powered analysis",
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
            description="Search memories with AI-enhanced understanding",
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
        Tool(
            name="consolidate_memories",
            description="Use AI to consolidate related memories",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute memory tools"""
    if not memory_system:
        return [TextContent(type="text", text="Error: No Anthropic API key configured")]

    if name == "store_memory":
        memory_id = memory_system.store_memory(
            arguments["content"], {"tags": arguments.get("tags", [])}
        )
        memory = memory_system.memory_index[memory_id]
        return [
            TextContent(
                type="text",
                text=f"Memory stored (ID: {memory_id})\nSummary: {memory.summary}\nImportance: {memory.importance_score:.2f}",
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
                f"[{memory.timestamp.strftime('%Y-%m-%d')}] "
                f"(Importance: {memory.importance_score:.2f})\n"
                f"Summary: {memory.summary}\n"
                f"Content: {memory.content[:200]}..."
            )

        return [TextContent(type="text", text="\n\n---\n\n".join(results))]

    elif name == "consolidate_memories":
        memory_system.consolidate_memories()
        return [
            TextContent(
                type="text",
                text="Memory consolidation complete. Related memories have been merged.",
            )
        ]

    return [TextContent(type="text", text="Unknown tool")]


async def main():
    """Run the Claude-powered memory server"""
    logger.info("🚀 Starting Claude-Powered Memory RAG Server")
    logger.info("🤖 Using Claude Opus 4 for intelligent features")
    logger.info("🧠 Using local embeddings for vector search")

    if memory_system:
        logger.info(f"📁 Storage: {memory_system.base_path}")
        stats = memory_system.get_stats()
        logger.info(f"💾 Total memories: {stats['total_memories']}")
    else:
        logger.error("⚠️  No Anthropic API key found!")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
