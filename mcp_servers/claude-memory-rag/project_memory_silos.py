#!/usr/bin/env python3
"""
Project-Based Memory Silo System
Organizes and isolates memories by project context
"""

import os
import sys
import json
import asyncio
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
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))
from server_full_featured import FullFeaturedMemoryRAG


class ProjectMemorySilo:
    """Manages project-specific memory silos"""

    def __init__(self):
        self.base_memory = FullFeaturedMemoryRAG()

        # Project definitions
        self.projects = {
            "JARVIS": {
                "path": "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM",
                "description": "Main JARVIS AI system",
                "keywords": ["jarvis", "ai", "ecosystem", "core", "swarm"],
                "memory_namespace": "jarvis_main",
            },
            "claude-memory": {
                "path": "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/mcp_servers/claude-memory-rag",
                "description": "Claude memory RAG system",
                "keywords": ["memory", "rag", "mcp", "langchain", "mem0"],
                "memory_namespace": "claude_memory",
            },
            "browser-control": {
                "path": "/Users/andreprofitt/browser-mcp-server",
                "description": "Browser automation MCP server",
                "keywords": ["browser", "playwright", "automation", "web"],
                "memory_namespace": "browser_mcp",
            },
            "desktop-automation": {
                "path": "/Users/andreprofitt/desktop-automation-server",
                "description": "Desktop automation server",
                "keywords": ["desktop", "automation", "applescript", "control"],
                "memory_namespace": "desktop_auto",
            },
        }

        # Current project context
        self.current_project = None

        # Project-specific memory stores (isolated)
        self.project_memories = {}
        self._initialize_project_stores()

    def _initialize_project_stores(self):
        """Initialize separate memory stores for each project"""
        for project_id, project_info in self.projects.items():
            self.project_memories[project_id] = {
                "conversations": [],
                "patterns": [],
                "context": project_info,
                "stats": {"total_memories": 0, "last_accessed": None},
            }

    def detect_project_context(
        self, content: str, metadata: Optional[Dict] = None
    ) -> str:
        """Automatically detect which project context we're in"""
        content_lower = content.lower()

        # Check metadata first
        if metadata and "project" in metadata:
            return metadata["project"]

        # Score each project based on keyword matches
        project_scores = {}

        for project_id, project_info in self.projects.items():
            score = 0

            # Check keywords
            for keyword in project_info["keywords"]:
                if keyword in content_lower:
                    score += 10

            # Check path mentions
            if project_info["path"].lower() in content_lower:
                score += 20

            # Check project name
            if project_id.lower() in content_lower:
                score += 15

            project_scores[project_id] = score

        # Return project with highest score
        if project_scores:
            best_project = max(project_scores.items(), key=lambda x: x[1])
            if best_project[1] > 0:  # Only if we found matches
                return best_project[0]

        # Default to current project or general
        return self.current_project or "general"

    async def set_project_context(self, project_id: str):
        """Set the current project context"""
        if project_id in self.projects:
            self.current_project = project_id
            print(f"📁 Switched to project: {project_id}")

            # Update last accessed
            self.project_memories[project_id]["stats"][
                "last_accessed"
            ] = datetime.now().isoformat()

            # Store context switch in memory
            await self.base_memory.store_conversation(
                f"context_switch_{int(datetime.now().timestamp())}",
                [
                    {
                        "role": "system",
                        "content": f"Project context switched to: {project_id}",
                    }
                ],
                {
                    "type": "context_switch",
                    "project": project_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        else:
            print(f"⚠️  Unknown project: {project_id}")

    async def store_project_memory(
        self,
        conversation_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None,
        project_override: Optional[str] = None,
    ):
        """Store memory with project isolation"""

        # Determine project
        project_id = project_override or self.detect_project_context(
            " ".join([m.get("content", "") for m in messages]), metadata
        )

        # Add project namespace to metadata
        if metadata is None:
            metadata = {}

        metadata["project"] = project_id
        metadata["namespace"] = self.projects.get(project_id, {}).get(
            "memory_namespace", project_id
        )

        # Store in base memory with project tagging
        success = await self.base_memory.store_conversation(
            f"{project_id}_{conversation_id}", messages, metadata
        )

        # Update project-specific tracking
        if project_id in self.project_memories:
            self.project_memories[project_id]["conversations"].append(
                {
                    "id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "message_count": len(messages),
                }
            )
            self.project_memories[project_id]["stats"]["total_memories"] += 1

        print(f"💾 Stored in project silo: {project_id}")
        return success

    async def search_project_memories(
        self,
        query: str,
        project_id: Optional[str] = None,
        cross_project: bool = False,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories with project filtering"""

        # If cross_project is False, limit to specific project
        if not cross_project:
            project_id = project_id or self.current_project

            if project_id:
                # Modify query to include project context
                namespace = self.projects.get(project_id, {}).get(
                    "memory_namespace", project_id
                )
                enhanced_query = f"[{namespace}] {query}"
            else:
                enhanced_query = query
        else:
            enhanced_query = f"[ALL PROJECTS] {query}"

        # Search using base memory
        all_memories = await self.base_memory.recall_memories(
            enhanced_query, top_k=top_k * 2
        )

        # Filter by project if not cross-project
        if not cross_project and project_id:
            filtered_memories = []
            for memory in all_memories:
                mem_project = memory.get("metadata", {}).get("project")
                if mem_project == project_id:
                    filtered_memories.append(memory)

            return filtered_memories[:top_k]

        return all_memories[:top_k]

    async def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get summary of a project's memory silo"""
        if project_id not in self.projects:
            return {"error": "Unknown project"}

        # Search for project memories
        project_memories = await self.search_project_memories(
            "",  # Empty query to get all
            project_id=project_id,
            cross_project=False,
            top_k=100,
        )

        # Analyze with GPT-4
        if project_memories:
            context = "\n".join(
                [m.get("content", "")[:200] for m in project_memories[:10]]
            )
            analysis = await self.base_memory.analyze_with_gpt4(
                f"Summarize the key aspects of the {project_id} project based on these memories",
                project_memories[:10],
            )
        else:
            analysis = "No memories found for this project yet."

        return {
            "project": project_id,
            "info": self.projects[project_id],
            "stats": self.project_memories[project_id]["stats"],
            "memory_count": len(project_memories),
            "analysis": analysis,
        }

    async def migrate_existing_memories(self):
        """Migrate existing memories to project silos"""
        print("\n🔄 Migrating existing memories to project silos...")

        # Get all existing memories
        all_memories = await self.base_memory.recall_memories("", top_k=1000)

        migrated = 0
        for memory in all_memories:
            # Try to detect project
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})

            project_id = self.detect_project_context(content, metadata)

            if project_id != "general":
                # Update metadata with project info
                metadata["project"] = project_id
                metadata["namespace"] = self.projects.get(project_id, {}).get(
                    "memory_namespace", project_id
                )
                metadata["migrated"] = True

                migrated += 1

        print(f"✅ Migrated {migrated} memories to project silos")

    def get_all_project_stats(self) -> Dict[str, Any]:
        """Get statistics for all project silos"""
        stats = {}

        for project_id, project_data in self.project_memories.items():
            stats[project_id] = {
                "description": self.projects[project_id]["description"],
                "total_memories": project_data["stats"]["total_memories"],
                "last_accessed": project_data["stats"]["last_accessed"],
                "conversation_count": len(project_data["conversations"]),
            }

        return stats


class ProjectAwareMCPServer:
    """MCP Server with project-aware memory silos"""

    def __init__(self):
        self.silo = ProjectMemorySilo()
        self.start_time = datetime.now()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests with project awareness"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        response = {"jsonrpc": "2.0", "id": request_id}

        try:
            if method == "initialize":
                response["result"] = {
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {
                        "name": "claude-memory-projects",
                        "version": "4.0.0",
                    },
                }

            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "set_project_context",
                            "description": "Set the current project context",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "project_id": {
                                        "type": "string",
                                        "enum": list(self.silo.projects.keys()),
                                    }
                                },
                                "required": ["project_id"],
                            },
                        },
                        {
                            "name": "store_project_memory",
                            "description": "Store memory in project-specific silo",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "conversation_id": {"type": "string"},
                                    "messages": {"type": "array"},
                                    "metadata": {"type": "object"},
                                    "project_override": {"type": "string"},
                                },
                                "required": ["conversation_id", "messages"],
                            },
                        },
                        {
                            "name": "search_project_memories",
                            "description": "Search within project silo or cross-project",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "project_id": {"type": "string"},
                                    "cross_project": {
                                        "type": "boolean",
                                        "default": False,
                                    },
                                    "top_k": {"type": "integer", "default": 5},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "get_project_summary",
                            "description": "Get summary of a project's memory silo",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"project_id": {"type": "string"}},
                                "required": ["project_id"],
                            },
                        },
                        {
                            "name": "get_all_project_stats",
                            "description": "Get statistics for all project silos",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ]
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                if tool_name == "set_project_context":
                    await self.silo.set_project_context(tool_params.get("project_id"))
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"✅ Switched to project: {tool_params.get('project_id')}",
                            }
                        ]
                    }

                elif tool_name == "store_project_memory":
                    success = await self.silo.store_project_memory(
                        tool_params.get("conversation_id"),
                        tool_params.get("messages", []),
                        tool_params.get("metadata"),
                        tool_params.get("project_override"),
                    )
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"✅ Stored in project silo: {self.silo.current_project or 'auto-detected'}",
                            }
                        ]
                    }

                elif tool_name == "search_project_memories":
                    memories = await self.silo.search_project_memories(
                        tool_params.get("query"),
                        tool_params.get("project_id"),
                        tool_params.get("cross_project", False),
                        tool_params.get("top_k", 5),
                    )

                    if memories:
                        memory_texts = []
                        for i, mem in enumerate(memories):
                            project = mem.get("metadata", {}).get("project", "unknown")
                            memory_texts.append(
                                f"[{project}] Memory {i+1}:\n{mem['content'][:200]}..."
                            )
                        response["result"] = {
                            "content": [
                                {"type": "text", "text": "\n\n".join(memory_texts)}
                            ]
                        }
                    else:
                        response["result"] = {
                            "content": [{"type": "text", "text": "No memories found."}]
                        }

                elif tool_name == "get_project_summary":
                    summary = await self.silo.get_project_summary(
                        tool_params.get("project_id")
                    )
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": json.dumps(summary, indent=2)}
                        ]
                    }

                elif tool_name == "get_all_project_stats":
                    stats = self.silo.get_all_project_stats()
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
            response["error"] = {"code": -32603, "message": f"Internal error: {str(e)}"}

        return response

    async def start_server(self):
        """Start MCP server"""
        print("🧠 Claude Memory - Project Silo System", file=sys.stderr)
        print(
            "📁 Projects configured:",
            ", ".join(self.silo.projects.keys()),
            file=sys.stderr,
        )
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


async def main():
    """Test project silo system"""
    print("🧪 Testing Project Silo System")
    print("=" * 60)

    silo = ProjectMemorySilo()

    # Test project detection
    test_content = "Working on the JARVIS core system with OpenAI integration"
    detected = silo.detect_project_context(test_content)
    print(f"Detected project: {detected}")

    # Get all project stats
    stats = silo.get_all_project_stats()
    print(f"\nProject stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    # If run directly, start the server
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        server = ProjectAwareMCPServer()
        asyncio.run(server.start_server())
    else:
        # Run test
        asyncio.run(main())
