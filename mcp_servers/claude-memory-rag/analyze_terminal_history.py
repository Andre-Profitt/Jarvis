#!/usr/bin/env python3
"""
Terminal History Analyzer
Analyzes bash/zsh history and organizes by project
"""

import os
import sys
import re
import json
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import subprocess

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))
from server_full_featured import FullFeaturedMemoryRAG


class TerminalHistoryAnalyzer:
    """Analyze terminal history and organize by project"""

    def __init__(self):
        self.memory = FullFeaturedMemoryRAG()
        self.home = Path.home()

        # Project directories to track
        self.project_dirs = {
            "JARVIS": "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM",
            "claude-memory-rag": "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/mcp_servers/claude-memory-rag",
            "browser-mcp": "/Users/andreprofitt/browser-mcp-server",
            "desktop-automation": "/Users/andreprofitt/desktop-automation-server",
            # Add more projects as needed
        }

        # Command patterns to track
        self.command_patterns = {
            "git": r"^git\s+(\w+)",
            "python": r"^python[3]?\s+(.+\.py)",
            "npm": r"^npm\s+(\w+)",
            "docker": r"^docker\s+(\w+)",
            "cd": r"^cd\s+(.+)",
            "test": r"(test|pytest|jest)",
            "install": r"(pip|npm|yarn|brew)\s+install",
            "server": r"(server|serve|run|start)",
        }

        self.stats = defaultdict(
            lambda: {
                "total_commands": 0,
                "command_types": defaultdict(int),
                "files_accessed": set(),
                "common_patterns": [],
            }
        )

    def get_history_files(self):
        """Find all history files"""
        history_files = []

        # Common history file locations
        potential_files = [
            self.home / ".bash_history",
            self.home / ".zsh_history",
            self.home / ".history",
            self.home / ".local/share/fish/fish_history",
        ]

        for file in potential_files:
            if file.exists():
                history_files.append(file)
                print(f"📋 Found history file: {file}")

        return history_files

    def parse_zsh_history(self, content):
        """Parse ZSH history format"""
        commands = []

        # ZSH history format: ": timestamp:duration;command"
        pattern = r"^:\s*(\d+):(\d+);(.+)$"

        for line in content.split("\n"):
            match = re.match(pattern, line)
            if match:
                timestamp, duration, command = match.groups()
                commands.append(
                    {
                        "command": command,
                        "timestamp": int(timestamp),
                        "duration": int(duration),
                    }
                )
            elif line.strip() and not line.startswith(":"):
                # Plain command without timestamp
                commands.append(
                    {"command": line.strip(), "timestamp": None, "duration": None}
                )

        return commands

    def parse_bash_history(self, content):
        """Parse bash history format"""
        commands = []

        for line in content.split("\n"):
            line = line.strip()
            if line:
                # Check if line has timestamp (some bash configs add it)
                if line.startswith("#") and line[1:].isdigit():
                    continue  # Skip timestamp lines

                commands.append({"command": line, "timestamp": None, "duration": None})

        return commands

    def determine_project(self, command):
        """Determine which project a command belongs to"""
        # Check for cd commands
        cd_match = re.match(r"^cd\s+(.+)", command)
        if cd_match:
            path = cd_match.group(1)
            for project, project_path in self.project_dirs.items():
                if project_path in path or path in project_path:
                    return project

        # Check for file paths in commands
        for project, project_path in self.project_dirs.items():
            if project_path in command:
                return project
            # Check for project name in command
            if project.lower() in command.lower():
                return project

        # Default to general
        return "general"

    def analyze_command(self, command, project):
        """Analyze a single command"""
        # Increment total
        self.stats[project]["total_commands"] += 1

        # Categorize command type
        for cmd_type, pattern in self.command_patterns.items():
            if re.search(pattern, command, re.IGNORECASE):
                self.stats[project]["command_types"][cmd_type] += 1

        # Extract file names
        py_files = re.findall(r"(\w+\.py)", command)
        for file in py_files:
            self.stats[project]["files_accessed"].add(file)

        # Store common patterns
        if any(keyword in command for keyword in ["test", "run", "server", "install"]):
            if command not in self.stats[project]["common_patterns"]:
                self.stats[project]["common_patterns"].append(command)

    async def analyze_history(self):
        """Analyze all history files"""
        print("\n🔍 Analyzing terminal history...")

        history_files = self.get_history_files()
        all_commands = []

        for history_file in history_files:
            try:
                with open(history_file, "r", errors="ignore") as f:
                    content = f.read()

                # Parse based on file type
                if "zsh" in history_file.name:
                    commands = self.parse_zsh_history(content)
                else:
                    commands = self.parse_bash_history(content)

                all_commands.extend(commands)
                print(f"✅ Parsed {len(commands)} commands from {history_file.name}")

            except Exception as e:
                print(f"❌ Error reading {history_file}: {e}")

        # Analyze commands by project
        print(f"\n📊 Analyzing {len(all_commands)} total commands...")

        for cmd_data in all_commands:
            command = cmd_data["command"]
            project = self.determine_project(command)
            self.analyze_command(command, project)

        return all_commands

    async def store_project_histories(self):
        """Store analyzed histories in memory by project"""
        print("\n💾 Storing project histories...")

        for project, stats in self.stats.items():
            if stats["total_commands"] > 0:
                # Convert sets to lists for JSON serialization
                stats_copy = dict(stats)
                stats_copy["files_accessed"] = list(stats["files_accessed"])
                stats_copy["command_types"] = dict(stats["command_types"])

                # Get top 20 common patterns
                stats_copy["common_patterns"] = stats_copy["common_patterns"][:20]

                # Store in memory
                await self.memory.store_conversation(
                    f"terminal_history_{project}_{int(datetime.now().timestamp())}",
                    [
                        {
                            "role": "system",
                            "content": f"Terminal History Analysis for {project}",
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(stats_copy, indent=2),
                        },
                    ],
                    {
                        "project": project,
                        "type": "terminal_history",
                        "category": "development_history",
                        "total_commands": stats["total_commands"],
                    },
                )

                print(
                    f"✅ Stored history for {project}: {stats['total_commands']} commands"
                )

    async def run_analysis(self):
        """Run the complete analysis"""
        print("🖥️  Terminal History Analysis")
        print("=" * 60)

        # Analyze history
        await self.analyze_history()

        # Store results
        await self.store_project_histories()

        # Print summary
        print("\n📊 Summary by Project:")
        print("-" * 40)

        for project, stats in sorted(
            self.stats.items(), key=lambda x: x[1]["total_commands"], reverse=True
        ):
            if stats["total_commands"] > 0:
                print(f"\n{project}:")
                print(f"  Total commands: {stats['total_commands']}")
                print(f"  Command types: {dict(stats['command_types'])}")
                print(f"  Files accessed: {len(stats['files_accessed'])}")
                print(f"  Common patterns: {len(stats['common_patterns'])}")


async def main():
    analyzer = TerminalHistoryAnalyzer()
    await analyzer.run_analysis()


if __name__ == "__main__":
    asyncio.run(main())
