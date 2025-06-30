#!/usr/bin/env python3
"""
Index JARVIS Codebase into Claude's Memory
This gives Claude deep understanding of the entire JARVIS ecosystem
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from server import ClaudeMemoryRAG


class JARVISIndexer:
    def __init__(self):
        self.memory = ClaudeMemoryRAG()
        self.jarvis_root = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM")
        self.indexed_count = 0

    async def index_codebase(self):
        """Index entire JARVIS codebase"""
        print("🔍 Indexing JARVIS Codebase into Claude's Memory...")
        print("=" * 50)

        # Core components to prioritize
        priority_dirs = ["core", "mcp_servers", "deployment", "tools"]

        for dir_name in priority_dirs:
            dir_path = self.jarvis_root / dir_name
            if dir_path.exists():
                await self.index_directory(dir_path)

        # Index other Python files
        for py_file in self.jarvis_root.rglob("*.py"):
            if not any(priority in str(py_file) for priority in priority_dirs):
                await self.index_file(py_file)

        # Index documentation
        for md_file in self.jarvis_root.rglob("*.md"):
            await self.index_documentation(md_file)

        # Index configuration
        config_files = ["config.yaml", "requirements.txt", "Dockerfile"]
        for config in config_files:
            config_path = self.jarvis_root / config
            if config_path.exists():
                await self.index_file(config_path)

        # Sync to GCS
        print("\n📤 Syncing to Google Cloud Storage...")
        await self.memory.sync_to_gcs()

        print(f"\n✅ Indexed {self.indexed_count} files into Claude's memory!")

        # Show memory stats
        stats = self.memory.get_memory_stats()
        print(f"\n📊 Memory Statistics:")
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Code files: {stats['memory_types'].get('code_understanding', 0)}")
        print(f"   Knowledge docs: {stats['memory_types'].get('project_knowledge', 0)}")

    async def index_directory(self, dir_path: Path):
        """Index all Python files in a directory"""
        print(f"\n📁 Indexing {dir_path.name}/...")

        for py_file in dir_path.rglob("*.py"):
            await self.index_file(py_file)

    async def index_file(self, file_path: Path):
        """Index a single code file"""
        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Skip empty or very small files
            if len(content) < 50:
                return

            # Generate analysis based on content
            analysis = self.analyze_code(file_path, content)

            # Store in memory
            relative_path = file_path.relative_to(self.jarvis_root)
            success = await self.memory.store_code_understanding(
                file_path=str(relative_path),
                code_content=content,
                analysis=analysis,
                metadata={
                    "component": self.identify_component(file_path),
                    "file_size": len(content),
                    "language": file_path.suffix[1:],
                },
            )

            if success:
                self.indexed_count += 1
                print(f"   ✅ {relative_path}")
            else:
                print(f"   ❌ Failed: {relative_path}")

        except Exception as e:
            print(f"   ⚠️  Error indexing {file_path}: {e}")

    async def index_documentation(self, doc_path: Path):
        """Index documentation files"""
        try:
            content = doc_path.read_text(encoding="utf-8", errors="ignore")

            # Store as project knowledge
            relative_path = doc_path.relative_to(self.jarvis_root)

            # For knowledge base docs, store differently
            collection_name = "project_knowledge"
            if "JARVIS-KNOWLEDGE" in str(doc_path):
                # These are especially important
                analysis = f"JARVIS Knowledge Base: {doc_path.stem}"
            else:
                analysis = f"Documentation: {doc_path.stem}"

            # Store in project_knowledge collection
            embedding = self.memory.embedder.encode(content[:1000]).tolist()

            self.memory.collections["project_knowledge"].add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[
                    {
                        "file_path": str(relative_path),
                        "doc_type": "markdown",
                        "title": doc_path.stem,
                    }
                ],
                ids=[f"doc_{doc_path.stem}"],
            )

            self.indexed_count += 1
            print(f"   📄 {relative_path}")

        except Exception as e:
            print(f"   ⚠️  Error indexing doc {doc_path}: {e}")

    def analyze_code(self, file_path: Path, content: str) -> str:
        """Generate analysis of code file"""
        # Extract key information
        lines = content.split("\n")

        # Find classes and functions
        classes = [line.strip() for line in lines if line.strip().startswith("class ")]
        functions = [line.strip() for line in lines if line.strip().startswith("def ")]

        # Extract docstring if present
        docstring = ""
        if '"""' in content:
            start = content.find('"""') + 3
            end = content.find('"""', start)
            if end > start:
                docstring = content[start:end].strip()

        # Build analysis
        analysis_parts = []

        if docstring:
            analysis_parts.append(f"Purpose: {docstring[:200]}")

        if classes:
            analysis_parts.append(
                f"Classes: {', '.join(c.split('(')[0].replace('class ', '') for c in classes[:3])}"
            )

        if functions:
            analysis_parts.append(
                f"Functions: {', '.join(f.split('(')[0].replace('def ', '') for f in functions[:5])}"
            )

        # Identify key patterns
        if "neural" in content.lower():
            analysis_parts.append("Component: Neural processing")
        if "quantum" in content.lower():
            analysis_parts.append("Component: Quantum optimization")
        if "self_healing" in content.lower() or "self-healing" in content.lower():
            analysis_parts.append("Component: Self-healing system")
        if "mcp" in content.lower():
            analysis_parts.append("Component: MCP integration")

        return " | ".join(analysis_parts) if analysis_parts else "Python code file"

    def identify_component(self, file_path: Path) -> str:
        """Identify which JARVIS component this file belongs to"""
        path_str = str(file_path).lower()

        if "neural" in path_str:
            return "neural_resource_manager"
        elif "self_healing" in path_str or "self-healing" in path_str:
            return "self_healing_system"
        elif "quantum" in path_str:
            return "quantum_optimization"
        elif "llm" in path_str or "research" in path_str:
            return "llm_research"
        elif "mcp" in path_str:
            return "mcp_integration"
        elif "core" in path_str:
            return "core_system"
        else:
            return "general"


async def main():
    """Run the indexing process"""
    print("🚀 JARVIS Codebase Indexer")
    print("This will give Claude deep understanding of JARVIS\n")

    # Check for GCS credentials
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        default_creds = Path.home() / ".gcs/jarvis-credentials.json"
        if default_creds.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_creds)
        else:
            print("⚠️  Warning: Google Cloud credentials not found!")
            print(
                "Set GOOGLE_APPLICATION_CREDENTIALS or place key at ~/.gcs/jarvis-credentials.json"
            )
            return

    # Run indexer
    indexer = JARVISIndexer()
    await indexer.index_codebase()

    print("\n🧠 Claude now has deep understanding of JARVIS!")
    print("You can ask me about any component, file, or concept!")


if __name__ == "__main__":
    asyncio.run(main())
