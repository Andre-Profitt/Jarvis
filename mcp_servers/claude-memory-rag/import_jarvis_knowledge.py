#!/usr/bin/env python3
"""
Import JARVIS Project Knowledge into Memory System
Analyzes the entire JARVIS ecosystem and stores it in memory
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
import hashlib
import subprocess

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))
from server_full_featured import FullFeaturedMemoryRAG


class JARVISKnowledgeImporter:
    """Import JARVIS ecosystem knowledge into memory"""

    def __init__(self):
        self.memory = FullFeaturedMemoryRAG()
        self.jarvis_path = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM")
        self.stats = {"files_processed": 0, "memories_created": 0, "errors": 0}

    async def import_jarvis_structure(self):
        """Import project structure and architecture"""
        print("📁 Importing JARVIS project structure...")

        # Core directories to analyze
        core_dirs = [
            "core",
            "integrations",
            "web",
            "models",
            "assistants",
            "mcp_servers",
            "memory",
            "swarm",
        ]

        structure_info = {
            "project": "JARVIS-ECOSYSTEM",
            "type": "project_structure",
            "directories": {},
        }

        for dir_name in core_dirs:
            dir_path = self.jarvis_path / dir_name
            if dir_path.exists():
                # Get file count and important files
                py_files = list(dir_path.rglob("*.py"))
                structure_info["directories"][dir_name] = {
                    "path": str(dir_path),
                    "python_files": len(py_files),
                    "key_files": [f.name for f in py_files[:5]],  # First 5 files
                }

        # Store structure knowledge
        await self.memory.store_conversation(
            f"jarvis_structure_{int(datetime.now().timestamp())}",
            [
                {"role": "system", "content": "JARVIS Project Structure Analysis"},
                {"role": "assistant", "content": json.dumps(structure_info, indent=2)},
            ],
            {
                "project": "JARVIS",
                "type": "architecture",
                "category": "project_structure",
            },
        )

        self.stats["memories_created"] += 1
        print(f"✅ Stored project structure")

    async def import_core_components(self):
        """Import core component documentation"""
        print("\n🧩 Importing core components...")

        # Key files to analyze
        key_files = [
            ("core/jarvis_core.py", "Core JARVIS engine"),
            ("core/memory_core.py", "Memory management system"),
            ("core/real_openai_integration.py", "OpenAI GPT-4 integration"),
            ("integrations/gemini_integration.py", "Google Gemini integration"),
            ("web/app.py", "Web interface"),
            ("swarm/swarm_system.py", "Multi-agent swarm system"),
            ("mcp_servers/jarvis_mcp.py", "MCP server integration"),
        ]

        for file_path, description in key_files:
            full_path = self.jarvis_path / file_path
            if full_path.exists():
                try:
                    # Read file content
                    with open(full_path, "r") as f:
                        content = f.read()

                    # Extract docstrings and key functions
                    lines = content.split("\n")
                    docstring = ""
                    functions = []

                    for i, line in enumerate(lines):
                        if line.strip().startswith('"""') and i < 10:
                            # Get module docstring
                            for j in range(i + 1, min(i + 20, len(lines))):
                                if '"""' in lines[j]:
                                    break
                                docstring += lines[j] + "\n"

                        if line.strip().startswith("def ") or line.strip().startswith(
                            "async def "
                        ):
                            func_name = (
                                line.split("(")[0]
                                .replace("def ", "")
                                .replace("async ", "")
                                .strip()
                            )
                            functions.append(func_name)

                    # Store component knowledge
                    component_info = {
                        "file": file_path,
                        "description": description,
                        "docstring": docstring.strip(),
                        "key_functions": functions[:10],  # Top 10 functions
                        "file_size": len(content),
                        "line_count": len(lines),
                    }

                    await self.memory.store_conversation(
                        f"jarvis_component_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        [
                            {
                                "role": "system",
                                "content": f"JARVIS Component: {description}",
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(component_info, indent=2),
                            },
                        ],
                        {
                            "project": "JARVIS",
                            "type": "component",
                            "file": file_path,
                            "category": "core_system",
                        },
                    )

                    self.stats["memories_created"] += 1
                    self.stats["files_processed"] += 1
                    print(f"✅ Imported {file_path}")

                except Exception as e:
                    print(f"❌ Error with {file_path}: {e}")
                    self.stats["errors"] += 1

    async def import_api_keys_and_config(self):
        """Import API keys and configuration (safely)"""
        print("\n🔑 Importing configuration...")

        # Read .env file
        env_path = self.jarvis_path / ".env"
        if env_path.exists():
            config_info = {
                "apis_configured": [],
                "features_enabled": [],
                "settings": {},
            }

            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key = key.strip()

                        # Store key names only (not values) for security
                        if "API_KEY" in key or "TOKEN" in key:
                            config_info["apis_configured"].append(key)
                        elif key.startswith("ENABLE_"):
                            config_info["features_enabled"].append(key)
                        elif key in ["DEBUG", "ENVIRONMENT", "LOG_LEVEL"]:
                            config_info["settings"][key] = value.strip()

            await self.memory.store_conversation(
                "jarvis_configuration",
                [
                    {"role": "system", "content": "JARVIS Configuration Overview"},
                    {"role": "assistant", "content": json.dumps(config_info, indent=2)},
                ],
                {
                    "project": "JARVIS",
                    "type": "configuration",
                    "category": "system_config",
                },
            )

            self.stats["memories_created"] += 1
            print("✅ Imported configuration (keys hidden)")

    async def import_capabilities(self):
        """Import JARVIS capabilities and features"""
        print("\n🚀 Importing JARVIS capabilities...")

        capabilities = {
            "core_features": [
                "Multi-model AI integration (GPT-4, Claude, Gemini)",
                "Advanced memory system with 30TB storage",
                "Real-time voice synthesis (ElevenLabs)",
                "WebSocket real-time communication",
                "Multi-agent swarm intelligence",
                "MCP server integration",
                "Code analysis and generation",
                "Family protection system",
            ],
            "integrations": [
                "OpenAI GPT-4",
                "Google Gemini (2M context)",
                "Claude Desktop",
                "ElevenLabs voice",
                "Google Cloud Storage",
                "Redis caching",
                "PostgreSQL/SQLite",
                "Web interface",
            ],
            "memory_features": [
                "Mem0 with OpenAI",
                "LangChain RAG",
                "ChromaDB vectors",
                "Google Cloud backup",
                "Conversation analysis",
                "Pattern learning",
            ],
        }

        await self.memory.store_conversation(
            "jarvis_capabilities",
            [
                {"role": "system", "content": "JARVIS System Capabilities"},
                {"role": "assistant", "content": json.dumps(capabilities, indent=2)},
            ],
            {"project": "JARVIS", "type": "capabilities", "category": "features"},
        )

        self.stats["memories_created"] += 1
        print("✅ Imported capabilities list")

    async def run_import(self):
        """Run the complete import process"""
        print("🤖 Starting JARVIS Knowledge Import")
        print("=" * 60)

        start_time = datetime.now()

        # Import all components
        await self.import_jarvis_structure()
        await self.import_core_components()
        await self.import_api_keys_and_config()
        await self.import_capabilities()

        # Final stats
        duration = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("✅ JARVIS Knowledge Import Complete!")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Memories created: {self.stats['memories_created']}")
        print(f"   Errors: {self.stats['errors']}")
        print(f"   Duration: {duration:.2f} seconds")

        return self.stats


async def main():
    importer = JARVISKnowledgeImporter()
    await importer.run_import()


if __name__ == "__main__":
    asyncio.run(main())
