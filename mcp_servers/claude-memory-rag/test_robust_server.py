#!/usr/bin/env python3
"""Test the robust MCP server"""

import sys
import os

sys.path.append(
    "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/mcp_servers/claude-memory-rag"
)

# Set up environment
os.environ["OPENAI_API_KEY"] = (
    "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"
)
os.environ["MEM0_API_KEY"] = "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC"

from server_robust import RobustMemoryRAG

# Test initialization
print("Testing server initialization...")
try:
    memory = RobustMemoryRAG()
    stats = memory.get_stats()
    print("Server initialized successfully!")
    print(f"Active systems: {stats['systems_active']}")
    print(f"Features: {stats['features']}")
except:
    pass
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
