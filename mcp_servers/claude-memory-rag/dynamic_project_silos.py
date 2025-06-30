#!/usr/bin/env python3
"""
Dynamic Project Memory Silo System
Automatically creates silos for any discovered project
Integrates with unified project discovery
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional, Set
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
from unified_project_discovery import UnifiedProjectDiscovery


class DynamicProjectMemorySilo:
    """Dynamic project-based memory silos that adapt to discovered projects"""

    def __init__(self):
        self.base_memory = FullFeaturedMemoryRAG()
        self.discovery = UnifiedProjectDiscovery()

        # Dynamic project registry (loaded from discovery)
        self.projects = {}
        self.project_keywords = {}
        self.project_paths = {}

        # Current context
        self.current_project = None
        self.recent_projects = []  # Stack of recently accessed projects

        # Load existing project configuration
        self._load_project_config()

    def _load_project_config(self):
        """Load project configuration from stored memory"""
        config_file = Path.home() / ".claude_full_memory" / "project_config.json"

        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                self.projects = config.get("projects", {})
                self.project_keywords = config.get("keywords", {})
                self.project_paths = config.get("paths", {})
                print(f"📋 Loaded {len(self.projects)} existing projects")

    def _save_project_config(self):
        """Save project configuration"""
        config_file = Path.home() / ".claude_full_memory" / "project_config.json"
        config_file.parent.mkdir(exist_ok=True)

        config = {
            "projects": self.projects,
            "keywords": self.project_keywords,
            "paths": self.project_paths,
            "last_updated": datetime.now().isoformat(),
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    async def sync_with_discovery(self):
        """Sync with unified project discovery"""
        print("\n🔄 Syncing with project discovery...")

        # Run discovery
        discovered_projects = await self.discovery.run_discovery()

        # Update project registry
        new_projects = 0
        updated_projects = 0

        for project_name, project_info in discovered_projects.items():
            if project_name not in self.projects:
                new_projects += 1

                # Create project entry
                self.projects[project_name] = {
                    "name": project_name,
                    "namespace": f"proj_{hashlib.md5(project_name.encode()).hexdigest()[:8]}",
                    "info": project_info,
                    "created": datetime.now().isoformat(),
                    "stats": {"memory_count": 0, "last_accessed": None},
                }

                # Extract keywords
                keywords = self._extract_keywords(project_name, project_info)
                self.project_keywords[project_name] = keywords

                # Store paths
                if "path" in project_info:
                    self.project_paths[project_info["path"]] = project_name
                if "local_path" in project_info:
                    self.project_paths[project_info["local_path"]] = project_name

            else:
                # Update existing project info
                self.projects[project_name]["info"].update(project_info)
                updated_projects += 1

        # Save configuration
        self._save_project_config()

        print(f"✅ Sync complete: {new_projects} new, {updated_projects} updated")

        # Store sync event in memory
        await self.base_memory.store_conversation(
            f"project_sync_{int(datetime.now().timestamp())}",
            [
                {"role": "system", "content": "Project Discovery Sync"},
                {
                    "role": "assistant",
                    "content": f"Synced {len(discovered_projects)} projects: {new_projects} new, {updated_projects} updated",
                },
            ],
            {
                "type": "project_sync",
                "total_projects": len(self.projects),
                "new_projects": new_projects,
                "updated_projects": updated_projects,
            },
        )

    def _extract_keywords(self, project_name: str, project_info: Dict) -> List[str]:
        """Extract keywords from project info"""
        keywords = set()

        # Add project name variations
        keywords.add(project_name.lower())
        keywords.add(project_name.replace("-", " ").lower())
        keywords.add(project_name.replace("_", " ").lower())

        # Add from description
        if "description" in project_info:
            desc_words = project_info["description"].lower().split()
            keywords.update(word for word in desc_words if len(word) > 3)

        # Add from topics (GitHub)
        if "topics" in project_info:
            keywords.update(project_info["topics"])

        # Add language
        if "language" in project_info:
            keywords.add(project_info["language"].lower())

        # Add type
        if "type" in project_info:
            keywords.add(project_info["type"].replace("/", " "))

        return list(keywords)

    def detect_project_context(
        self, content: str, metadata: Optional[Dict] = None
    ) -> str:
        """Intelligently detect project context"""
        content_lower = content.lower()

        # Check explicit metadata
        if metadata and "project" in metadata:
            if metadata["project"] in self.projects:
                return metadata["project"]

        # Score each project
        project_scores = {}

        for project_name, keywords in self.project_keywords.items():
            score = 0

            # Check project name
            if project_name.lower() in content_lower:
                score += 20

            # Check keywords
            for keyword in keywords:
                if keyword in content_lower:
                    score += 5

            # Check paths
            project_info = self.projects[project_name]["info"]
            if "path" in project_info and project_info["path"] in content:
                score += 30
            if "github_url" in project_info and project_info["github_url"] in content:
                score += 25

            # Boost score for recently accessed projects
            if project_name in self.recent_projects:
                score += 10

            if score > 0:
                project_scores[project_name] = score

        # Return highest scoring project
        if project_scores:
            best_project = max(project_scores.items(), key=lambda x: x[1])
            if best_project[1] >= 10:  # Minimum confidence threshold
                return best_project[0]

        # Check file paths for any project match
        for path, project in self.project_paths.items():
            if path in content:
                return project

        # Default to current project or general
        return self.current_project or "general"

    async def set_project_context(self, project_name: str):
        """Set current project context"""
        if project_name not in self.projects:
            # Try to find by partial match
            matches = [p for p in self.projects if project_name.lower() in p.lower()]
            if matches:
                project_name = matches[0]
            else:
                print(f"⚠️  Unknown project: {project_name}")
                print(f"   Available: {', '.join(list(self.projects.keys())[:10])}...")
                return False

        self.current_project = project_name

        # Update recent projects stack
        if project_name in self.recent_projects:
            self.recent_projects.remove(project_name)
        self.recent_projects.insert(0, project_name)
        if len(self.recent_projects) > 10:
            self.recent_projects = self.recent_projects[:10]

        # Update stats
        self.projects[project_name]["stats"][
            "last_accessed"
        ] = datetime.now().isoformat()
        self._save_project_config()

        print(f"📁 Switched to project: {project_name}")

        # Get project info
        info = self.projects[project_name]["info"]
        print(f"   Type: {info.get('type', 'unknown')}")
        print(f"   Source: {info.get('source', 'unknown')}")
        if "path" in info:
            print(f"   Path: {info['path']}")
        if "github_url" in info:
            print(f"   GitHub: {info['github_url']}")

        return True

    async def store_project_memory(
        self,
        conversation_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None,
        project_override: Optional[str] = None,
    ):
        """Store memory with dynamic project detection"""

        # Determine project
        project_name = project_override or self.detect_project_context(
            " ".join([m.get("content", "") for m in messages]), metadata
        )

        # Ensure project exists
        if project_name != "general" and project_name not in self.projects:
            # Auto-create project entry
            print(f"🆕 Auto-creating project: {project_name}")
            self.projects[project_name] = {
                "name": project_name,
                "namespace": f"proj_{hashlib.md5(project_name.encode()).hexdigest()[:8]}",
                "info": {"source": "auto-detected"},
                "created": datetime.now().isoformat(),
                "stats": {
                    "memory_count": 0,
                    "last_accessed": datetime.now().isoformat(),
                },
            }
            self.project_keywords[project_name] = [project_name.lower()]
            self._save_project_config()

        # Add project metadata
        if metadata is None:
            metadata = {}

        metadata["project"] = project_name
        if project_name in self.projects:
            metadata["namespace"] = self.projects[project_name]["namespace"]
            self.projects[project_name]["stats"]["memory_count"] += 1

        # Store in base memory
        success = await self.base_memory.store_conversation(
            f"{project_name}_{conversation_id}", messages, metadata
        )

        print(f"💾 Stored in project: {project_name}")
        return success

    async def search_project_memories(
        self,
        query: str,
        project_filter: Optional[List[str]] = None,
        include_general: bool = True,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories with flexible project filtering"""

        # Build enhanced query
        if project_filter:
            # Search specific projects
            namespaces = []
            for project in project_filter:
                if project in self.projects:
                    namespaces.append(self.projects[project]["namespace"])

            if namespaces:
                enhanced_query = f"[{','.join(namespaces)}] {query}"
            else:
                enhanced_query = query
        else:
            # Search all or current project
            if self.current_project and not include_general:
                namespace = self.projects[self.current_project]["namespace"]
                enhanced_query = f"[{namespace}] {query}"
            else:
                enhanced_query = query

        # Search
        memories = await self.base_memory.recall_memories(
            enhanced_query, top_k=top_k * 2
        )

        # Filter results if needed
        if project_filter or (self.current_project and not include_general):
            filtered = []
            allowed_projects = project_filter or (
                [self.current_project] if not include_general else None
            )

            for memory in memories:
                mem_project = memory.get("metadata", {}).get("project")
                if (
                    not allowed_projects
                    or mem_project in allowed_projects
                    or (include_general and mem_project == "general")
                ):
                    filtered.append(memory)

            return filtered[:top_k]

        return memories[:top_k]

    async def get_project_insights(self, project_name: str) -> Dict[str, Any]:
        """Get AI-powered insights about a project"""
        if project_name not in self.projects:
            return {"error": f"Unknown project: {project_name}"}

        # Get project memories
        memories = await self.search_project_memories(
            "",  # Empty query
            project_filter=[project_name],
            include_general=False,
            top_k=20,
        )

        # Get project info
        project_info = self.projects[project_name]

        # Prepare context for GPT-4
        context = {
            "project_info": project_info["info"],
            "memory_count": len(memories),
            "keywords": self.project_keywords.get(project_name, []),
            "recent_discussions": [m.get("content", "")[:200] for m in memories[:5]],
        }

        # Get GPT-4 analysis
        analysis = await self.base_memory.analyze_with_gpt4(
            f"Provide insights about the {project_name} project based on the memory context",
            memories[:10],
        )

        return {
            "project": project_name,
            "info": project_info,
            "memory_count": len(memories),
            "keywords": self.project_keywords.get(project_name, []),
            "ai_insights": analysis,
            "last_accessed": project_info["stats"].get("last_accessed"),
            "related_projects": self._find_related_projects(project_name),
        }

    def _find_related_projects(self, project_name: str) -> List[str]:
        """Find projects related to the given project"""
        related = []

        if project_name not in self.projects:
            return related

        project_keywords = set(self.project_keywords.get(project_name, []))
        project_info = self.projects[project_name]["info"]

        for other_name, other_keywords in self.project_keywords.items():
            if other_name != project_name:
                # Check keyword overlap
                overlap = len(project_keywords.intersection(set(other_keywords)))
                if overlap >= 2:
                    related.append(other_name)
                    continue

                # Check same language
                if project_info.get("language") and project_info[
                    "language"
                ] == self.projects[other_name]["info"].get("language"):
                    related.append(other_name)

        return related[:5]  # Top 5 related

    def get_project_list(self, filter_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of all projects with optional filtering"""
        project_list = []

        for name, project in self.projects.items():
            # Apply filter
            if filter_by:
                if filter_by == "recent":
                    if name not in self.recent_projects[:10]:
                        continue
                elif filter_by == "github":
                    if "github_url" not in project["info"]:
                        continue
                elif filter_by == "local":
                    if (
                        "path" not in project["info"]
                        and "local_path" not in project["info"]
                    ):
                        continue
                elif filter_by.startswith("type:"):
                    project_type = filter_by.split(":", 1)[1]
                    if project["info"].get("type") != project_type:
                        continue
                elif filter_by.startswith("language:"):
                    language = filter_by.split(":", 1)[1]
                    if project["info"].get("language", "").lower() != language.lower():
                        continue

            project_list.append(
                {
                    "name": name,
                    "type": project["info"].get("type", "unknown"),
                    "source": project["info"].get("source", "unknown"),
                    "language": project["info"].get("language"),
                    "has_local": "path" in project["info"]
                    or "local_path" in project["info"],
                    "has_github": "github_url" in project["info"],
                    "memory_count": project["stats"]["memory_count"],
                    "last_accessed": project["stats"].get("last_accessed"),
                }
            )

        # Sort by last accessed
        project_list.sort(key=lambda x: x["last_accessed"] or "0", reverse=True)

        return project_list


class DynamicProjectMCPServer:
    """MCP Server with dynamic project discovery and management"""

    def __init__(self):
        self.silo = DynamicProjectMemorySilo()
        self.start_time = datetime.now()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        response = {"jsonrpc": "2.0", "id": request_id}

        try:
            if method == "initialize":
                # Run initial sync
                asyncio.create_task(self.silo.sync_with_discovery())

                response["result"] = {
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "claude-memory-unified", "version": "5.0.0"},
                }

            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "sync_projects",
                            "description": "Sync with all project sources (Claude, GitHub, local)",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "set_project",
                            "description": "Set current project context",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"project_name": {"type": "string"}},
                                "required": ["project_name"],
                            },
                        },
                        {
                            "name": "store_memory",
                            "description": "Store memory with automatic project detection",
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
                            "name": "search_memories",
                            "description": "Search memories with flexible project filtering",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "project_filter": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "include_general": {
                                        "type": "boolean",
                                        "default": True,
                                    },
                                    "top_k": {"type": "integer", "default": 5},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "get_project_insights",
                            "description": "Get AI-powered insights about a project",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"project_name": {"type": "string"}},
                                "required": ["project_name"],
                            },
                        },
                        {
                            "name": "list_projects",
                            "description": "List all discovered projects",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "filter": {
                                        "type": "string",
                                        "enum": [
                                            "all",
                                            "recent",
                                            "github",
                                            "local",
                                            "type:*",
                                            "language:*",
                                        ],
                                    }
                                },
                            },
                        },
                        {
                            "name": "get_current_context",
                            "description": "Get current project context and recent projects",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ]
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                if tool_name == "sync_projects":
                    await self.silo.sync_with_discovery()
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"✅ Project sync complete! Found {len(self.silo.projects)} projects",
                            }
                        ]
                    }

                elif tool_name == "set_project":
                    success = await self.silo.set_project_context(
                        tool_params.get("project_name")
                    )
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"{'✅' if success else '❌'} {'Set' if success else 'Failed to set'} project: {tool_params.get('project_name')}",
                            }
                        ]
                    }

                elif tool_name == "store_memory":
                    success = await self.silo.store_project_memory(
                        tool_params.get("conversation_id"),
                        tool_params.get("messages", []),
                        tool_params.get("metadata"),
                        tool_params.get("project_override"),
                    )
                    detected_project = self.silo.current_project or "auto-detected"
                    response["result"] = {
                        "content": [
                            {
                                "type": "text",
                                "text": f"✅ Stored in project: {detected_project}",
                            }
                        ]
                    }

                elif tool_name == "search_memories":
                    memories = await self.silo.search_project_memories(
                        tool_params.get("query"),
                        tool_params.get("project_filter"),
                        tool_params.get("include_general", True),
                        tool_params.get("top_k", 5),
                    )

                    if memories:
                        memory_texts = []
                        for i, mem in enumerate(memories):
                            project = mem.get("metadata", {}).get("project", "general")
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

                elif tool_name == "get_project_insights":
                    insights = await self.silo.get_project_insights(
                        tool_params.get("project_name")
                    )
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": json.dumps(insights, indent=2)}
                        ]
                    }

                elif tool_name == "list_projects":
                    projects = self.silo.get_project_list(
                        tool_params.get("filter", "all")
                    )

                    # Format as table
                    text = f"📁 Projects ({len(projects)} total):\n\n"
                    text += (
                        "Name | Type | Source | Language | Local | GitHub | Memories\n"
                    )
                    text += "-" * 70 + "\n"

                    for p in projects[:20]:  # Show top 20
                        text += f"{p['name'][:20]:<20} | "
                        text += f"{p['type'][:10]:<10} | "
                        text += f"{p['source'][:8]:<8} | "
                        text += f"{p.get('language', 'N/A')[:8]:<8} | "
                        text += f"{'✓' if p['has_local'] else '✗':<5} | "
                        text += f"{'✓' if p['has_github'] else '✗':<6} | "
                        text += f"{p['memory_count']}\n"

                    if len(projects) > 20:
                        text += f"\n... and {len(projects) - 20} more projects"

                    response["result"] = {"content": [{"type": "text", "text": text}]}

                elif tool_name == "get_current_context":
                    context = {
                        "current_project": self.silo.current_project,
                        "recent_projects": self.silo.recent_projects[:5],
                        "total_projects": len(self.silo.projects),
                    }
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": json.dumps(context, indent=2)}
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
        print("🧠 Claude Memory - Unified Project System", file=sys.stderr)
        print(
            "🔄 Auto-discovering projects from Claude, GitHub, and local...",
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


if __name__ == "__main__":
    # If run directly, start the server
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        server = DynamicProjectMCPServer()
        asyncio.run(server.start_server())
    else:
        # Run test
        async def test():
            silo = DynamicProjectMemorySilo()
            await silo.sync_with_discovery()
            print(f"\n✅ Loaded {len(silo.projects)} projects")
            print(f"Recent: {silo.recent_projects[:5]}")

        asyncio.run(test())
