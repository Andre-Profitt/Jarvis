#!/usr/bin/env python3
"""
Unified Project Memory System
Auto-discovers projects from Claude Desktop, GitHub, and code history
Creates a unified knowledge base across all sources
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
import hashlib
import re

# GitHub integration
try:
    from github import Github

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    print("⚠️  PyGithub not installed, GitHub sync disabled", file=sys.stderr)

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))
from server_full_featured import FullFeaturedMemoryRAG


class UnifiedProjectDiscovery:
    """Discovers and unifies all projects across Claude, GitHub, and local directories"""

    def __init__(self):
        self.memory = FullFeaturedMemoryRAG()

        # Claude Desktop paths
        self.claude_projects_dir = Path.home() / "Documents" / "Claude" / "Projects"
        self.claude_app_support = (
            Path.home() / "Library" / "Application Support" / "Claude"
        )

        # Common project directories
        self.project_search_paths = [
            Path.home() / "Projects",
            Path.home() / "Documents" / "Projects",
            Path.home() / "Documents" / "Claude" / "Projects",
            Path.home() / "Developer",
            Path.home() / "Code",
            Path.home() / "CloudAI",
            Path.home() / "Desktop",
            Path("/Users/andreprofitt/CloudAI"),  # Your specific path
        ]

        # GitHub token from environment or file
        self.github_token = self._get_github_token()

        # Discovered projects
        self.all_projects = {}
        self.project_relationships = {}

    def _get_github_token(self):
        """Get GitHub token from various sources"""
        # Check environment
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            return token

        # Check JARVIS .env
        env_path = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/.env")
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("GITHUB_TOKEN="):
                        return line.split("=", 1)[1].strip()

        # Check Claude config
        config_path = self.claude_app_support / "claude_desktop_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                github_config = config.get("mcpServers", {}).get("github", {})
                return github_config.get("env", {}).get("GITHUB_PERSONAL_ACCESS_TOKEN")

        return None

    async def discover_claude_projects(self):
        """Discover all projects in Claude Desktop"""
        print("\n🔍 Discovering Claude Desktop projects...")

        claude_projects = {}

        # Check Claude project directory
        if self.claude_projects_dir.exists():
            for project_dir in self.claude_projects_dir.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    project_info = {
                        "name": project_dir.name,
                        "path": str(project_dir),
                        "source": "claude_desktop",
                        "type": self._detect_project_type(project_dir),
                        "last_modified": datetime.fromtimestamp(
                            project_dir.stat().st_mtime
                        ).isoformat(),
                    }

                    # Look for project metadata
                    metadata_file = project_dir / ".claude" / "project.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            project_info["claude_metadata"] = json.load(f)

                    claude_projects[project_dir.name] = project_info
                    print(f"  📁 Found: {project_dir.name}")

        # Check Claude code history
        code_history = await self._parse_claude_code_history()
        for project_name, history_info in code_history.items():
            if project_name in claude_projects:
                claude_projects[project_name]["code_history"] = history_info
            else:
                claude_projects[project_name] = {
                    "name": project_name,
                    "source": "claude_history",
                    "code_history": history_info,
                }

        self.all_projects.update(claude_projects)
        return claude_projects

    async def _parse_claude_code_history(self):
        """Parse Claude's code history to find projects"""
        print("\n📜 Parsing Claude code history...")

        code_history = {}

        # Look for Claude conversation history
        claude_data_dir = self.claude_app_support / "conversations"
        if claude_data_dir.exists():
            for conv_file in claude_data_dir.glob("*.json"):
                try:
                    with open(conv_file, "r") as f:
                        conv_data = json.load(f)

                    # Extract project references from conversations
                    for message in conv_data.get("messages", []):
                        content = message.get("content", "")

                        # Look for file paths
                        file_paths = re.findall(r"[/~][\w/.-]+\.\w+", content)
                        for path in file_paths:
                            project = self._extract_project_from_path(path)
                            if project:
                                if project not in code_history:
                                    code_history[project] = {
                                        "files_mentioned": set(),
                                        "conversation_count": 0,
                                    }
                                code_history[project]["files_mentioned"].add(path)
                                code_history[project]["conversation_count"] += 1

                except Exception as e:
                    print(f"  ⚠️  Error parsing {conv_file}: {e}")

        # Convert sets to lists for JSON serialization
        for project in code_history:
            code_history[project]["files_mentioned"] = list(
                code_history[project]["files_mentioned"]
            )

        return code_history

    def _extract_project_from_path(self, path: str) -> Optional[str]:
        """Extract project name from a file path"""
        # Common project indicators in paths
        patterns = [
            r"/Projects/([^/]+)",
            r"/Code/([^/]+)",
            r"/CloudAI/([^/]+)",
            r"/([^/]+)/(?:src|lib|app|server)",
            r"/([^/]+)/package\.json",
            r"/([^/]+)/\.git",
        ]

        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                return match.group(1)

        return None

    async def discover_github_projects(self):
        """Discover all GitHub projects"""
        if not HAS_GITHUB or not self.github_token:
            print("\n⚠️  GitHub integration not available")
            return {}

        print("\n🐙 Discovering GitHub projects...")

        github_projects = {}

        try:
            g = Github(self.github_token)
            user = g.get_user()

            # Get user's repositories
            for repo in user.get_repos():
                project_info = {
                    "name": repo.name,
                    "source": "github",
                    "github_url": repo.html_url,
                    "description": repo.description,
                    "language": repo.language,
                    "topics": repo.get_topics(),
                    "stars": repo.stargazers_count,
                    "last_updated": repo.updated_at.isoformat(),
                    "default_branch": repo.default_branch,
                    "is_private": repo.private,
                }

                # Check if this project exists locally
                local_path = self._find_local_path(repo.name)
                if local_path:
                    project_info["local_path"] = str(local_path)
                    project_info["has_local_copy"] = True

                github_projects[repo.name] = project_info
                print(
                    f"  🔗 Found: {repo.name} {'(local)' if local_path else '(remote only)'}"
                )

        except Exception as e:
            print(f"  ❌ GitHub error: {e}")

        self.all_projects.update(github_projects)
        return github_projects

    def _find_local_path(self, project_name: str) -> Optional[Path]:
        """Find local path for a project"""
        for search_path in self.project_search_paths:
            if search_path.exists():
                # Direct match
                project_path = search_path / project_name
                if project_path.exists() and project_path.is_dir():
                    return project_path

                # Search subdirectories
                for subdir in search_path.iterdir():
                    if subdir.is_dir() and subdir.name == project_name:
                        return subdir

        return None

    async def discover_local_projects(self):
        """Discover all local projects not already found"""
        print("\n💻 Discovering local projects...")

        local_projects = {}
        already_found = set(self.all_projects.keys())

        for search_path in self.project_search_paths:
            if search_path.exists():
                for project_dir in search_path.iterdir():
                    if (
                        project_dir.is_dir()
                        and not project_dir.name.startswith(".")
                        and project_dir.name not in already_found
                    ):

                        # Check if it's a valid project
                        if self._is_valid_project(project_dir):
                            project_info = {
                                "name": project_dir.name,
                                "path": str(project_dir),
                                "source": "local",
                                "type": self._detect_project_type(project_dir),
                                "last_modified": datetime.fromtimestamp(
                                    project_dir.stat().st_mtime
                                ).isoformat(),
                            }

                            # Get git info if available
                            git_info = self._get_git_info(project_dir)
                            if git_info:
                                project_info["git"] = git_info

                            local_projects[project_dir.name] = project_info
                            already_found.add(project_dir.name)
                            print(
                                f"  💾 Found: {project_dir.name} ({project_info['type']})"
                            )

        self.all_projects.update(local_projects)
        return local_projects

    def _is_valid_project(self, path: Path) -> bool:
        """Check if a directory is a valid project"""
        # Project indicators
        indicators = [
            ".git",
            "package.json",
            "requirements.txt",
            "setup.py",
            "Cargo.toml",
            "go.mod",
            ".project",
            "README.md",
            "Makefile",
        ]

        for indicator in indicators:
            if (path / indicator).exists():
                return True

        # Check for source code
        code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"]
        for ext in code_extensions:
            if list(path.glob(f"*{ext}")):
                return True

        return False

    def _detect_project_type(self, path: Path) -> str:
        """Detect the type of project"""
        if (path / "package.json").exists():
            return "node/javascript"
        elif (path / "requirements.txt").exists() or (path / "setup.py").exists():
            return "python"
        elif (path / "Cargo.toml").exists():
            return "rust"
        elif (path / "go.mod").exists():
            return "go"
        elif (path / "pom.xml").exists():
            return "java/maven"
        elif (path / ".git").exists():
            return "git-repository"
        else:
            return "unknown"

    def _get_git_info(self, path: Path) -> Optional[Dict]:
        """Get git information for a project"""
        git_dir = path / ".git"
        if not git_dir.exists():
            return None

        try:
            # Get current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            current_branch = result.stdout.strip()

            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            remote_url = result.stdout.strip()

            # Get last commit
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H %s"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            last_commit = result.stdout.strip()

            return {
                "current_branch": current_branch,
                "remote_url": remote_url,
                "last_commit": last_commit,
            }

        except Exception:
            return None

    async def analyze_project_relationships(self):
        """Analyze relationships between projects"""
        print("\n🔗 Analyzing project relationships...")

        relationships = {}

        for project_name, project_info in self.all_projects.items():
            relationships[project_name] = {
                "dependencies": [],
                "dependents": [],
                "related": [],
            }

            # Check for dependencies
            if project_info.get("path"):
                path = Path(project_info["path"])

                # Node.js dependencies
                package_json = path / "package.json"
                if package_json.exists():
                    with open(package_json, "r") as f:
                        pkg = json.load(f)
                        deps = list(pkg.get("dependencies", {}).keys())
                        dev_deps = list(pkg.get("devDependencies", {}).keys())

                        # Check if any deps are our projects
                        for dep in deps + dev_deps:
                            if dep in self.all_projects:
                                relationships[project_name]["dependencies"].append(dep)

                # Python dependencies
                requirements = path / "requirements.txt"
                if requirements.exists():
                    with open(requirements, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                # Extract package name
                                pkg_name = re.split("[<>=!]", line)[0].strip()
                                if pkg_name in self.all_projects:
                                    relationships[project_name]["dependencies"].append(
                                        pkg_name
                                    )

        self.project_relationships = relationships
        return relationships

    async def create_unified_knowledge_base(self):
        """Create unified knowledge base from all discovered projects"""
        print("\n🧠 Creating unified knowledge base...")

        # Store overall project map
        await self.memory.store_conversation(
            f"unified_project_map_{int(datetime.now().timestamp())}",
            [
                {"role": "system", "content": "Unified Project Discovery Results"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "total_projects": len(self.all_projects),
                            "sources": {
                                "claude": len(
                                    [
                                        p
                                        for p in self.all_projects.values()
                                        if p["source"] == "claude_desktop"
                                    ]
                                ),
                                "github": len(
                                    [
                                        p
                                        for p in self.all_projects.values()
                                        if p["source"] == "github"
                                    ]
                                ),
                                "local": len(
                                    [
                                        p
                                        for p in self.all_projects.values()
                                        if p["source"] == "local"
                                    ]
                                ),
                            },
                            "project_list": list(self.all_projects.keys()),
                        },
                        indent=2,
                    ),
                },
            ],
            {
                "type": "project_map",
                "category": "unified_discovery",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Store individual project details
        for project_name, project_info in self.all_projects.items():
            # Create comprehensive project memory
            project_memory = {
                **project_info,
                "relationships": self.project_relationships.get(project_name, {}),
                "unified_id": hashlib.md5(project_name.encode()).hexdigest()[:8],
            }

            await self.memory.store_conversation(
                f"unified_project_{project_memory['unified_id']}",
                [
                    {
                        "role": "system",
                        "content": f"Unified Project Information: {project_name}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(project_memory, indent=2),
                    },
                ],
                {
                    "project": project_name,
                    "type": "unified_project_info",
                    "source": project_info["source"],
                    "has_local": "path" in project_info or "local_path" in project_info,
                    "has_github": "github_url" in project_info,
                    "has_claude": project_info["source"] == "claude_desktop"
                    or "code_history" in project_info,
                },
            )

            print(f"  ✅ Stored: {project_name}")

        # Create project index for fast lookup
        project_index = {
            "projects": {},
            "by_type": {},
            "by_language": {},
            "by_source": {},
        }

        for name, info in self.all_projects.items():
            # Index by name
            project_index["projects"][name] = {
                "unified_id": hashlib.md5(name.encode()).hexdigest()[:8],
                "sources": [info["source"]],
                "type": info.get("type", "unknown"),
                "language": info.get("language", "unknown"),
            }

            # Index by type
            project_type = info.get("type", "unknown")
            if project_type not in project_index["by_type"]:
                project_index["by_type"][project_type] = []
            project_index["by_type"][project_type].append(name)

            # Index by language
            language = info.get("language", "unknown")
            if language not in project_index["by_language"]:
                project_index["by_language"][language] = []
            project_index["by_language"][language].append(name)

            # Index by source
            source = info["source"]
            if source not in project_index["by_source"]:
                project_index["by_source"][source] = []
            project_index["by_source"][source].append(name)

        # Store the index
        await self.memory.store_conversation(
            "unified_project_index",
            [
                {"role": "system", "content": "Unified Project Index for Fast Lookup"},
                {"role": "assistant", "content": json.dumps(project_index, indent=2)},
            ],
            {
                "type": "project_index",
                "category": "unified_discovery",
                "total_projects": len(self.all_projects),
            },
        )

        print(
            f"\n✅ Unified knowledge base created with {len(self.all_projects)} projects!"
        )

    async def run_discovery(self):
        """Run complete project discovery"""
        print("🚀 Unified Project Discovery System")
        print("=" * 60)

        start_time = datetime.now()

        # Discover from all sources
        await self.discover_claude_projects()
        await self.discover_github_projects()
        await self.discover_local_projects()

        # Analyze relationships
        await self.analyze_project_relationships()

        # Create unified knowledge base
        await self.create_unified_knowledge_base()

        # Summary
        duration = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("📊 Discovery Summary:")
        print(f"   Total projects found: {len(self.all_projects)}")
        print(
            f"   Claude projects: {len([p for p in self.all_projects.values() if 'claude' in p.get('source', '')])}"
        )
        print(
            f"   GitHub projects: {len([p for p in self.all_projects.values() if p['source'] == 'github'])}"
        )
        print(
            f"   Local projects: {len([p for p in self.all_projects.values() if p['source'] == 'local'])}"
        )
        print(f"   Duration: {duration:.2f} seconds")

        # Show project types
        print("\n📁 Project Types:")
        type_counts = {}
        for project in self.all_projects.values():
            ptype = project.get("type", "unknown")
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        for ptype, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   {ptype}: {count}")

        return self.all_projects


async def main():
    discovery = UnifiedProjectDiscovery()
    await discovery.run_discovery()


if __name__ == "__main__":
    # Install PyGithub if needed
    try:
        import github
    except ImportError:
        print("📦 Installing PyGithub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyGithub"])

    asyncio.run(main())
