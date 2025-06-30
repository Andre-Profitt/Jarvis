#!/usr/bin/env python3
"""
JARVIS Memory Wrapper - Easy access to enhanced memory system
"""

from pathlib import Path
import sys

# Add paths for imports
jarvis_path = Path.home() / ".jarvis_memory" / "cloud_sync"
if str(jarvis_path) not in sys.path:
    sys.path.insert(0, str(jarvis_path))

from redis_cache_layer import MemoryCache


class JarvisMemory:
    def __init__(self):
        self.cache = MemoryCache(
            redis_config={"host": "localhost", "port": 6379},
            pg_config={
                "host": "localhost",
                "port": 5432,
                "database": "unified_memory",
                "user": "andreprofitt",
            },
        )

    def remember(self, content, project="jarvis"):
        """Add a memory"""
        return self.cache.add_memory(project, content)

    def recall(self, query, project=None, limit=10):
        """Search memories"""
        return self.cache.search_memories(query, project_name=project, limit=limit)

    def reflect(self):
        """Get consciousness state"""
        return self.cache.get_consciousness_state()

    def get_projects(self):
        """List all projects"""
        with self.cache.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, memory_count, insight_count, last_activity
                    FROM project_stats
                    ORDER BY last_activity DESC NULLS LAST
                """
                )
                return [dict(row) for row in cur.fetchall()]

    def get_hot_memories(self, limit=10):
        """Get most accessed memories"""
        return self.cache.get_hot_memories(limit=limit)

    def update_consciousness(self, state_data):
        """Update consciousness state"""
        self.cache.update_consciousness_state(state_data)

    def get_stats(self):
        """Get system statistics"""
        patterns = self.cache.analyze_access_patterns()
        total_memories = 0

        with self.cache.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories")
                total_memories = cur.fetchone()["count"]

        return {
            "total_memories": total_memories,
            "cache_hit_rate": patterns["cache_stats"]["hit_rate"],
            "hot_memories": len(patterns["hot_memories"]),
            "recent_searches": len(patterns["recent_searches"]),
        }


# Convenience instance
memory = JarvisMemory()

# Quick test when imported
if __name__ != "__main__":
    try:
        stats = memory.get_stats()
        print(f"✓ JARVIS Memory System Active: {stats['total_memories']} memories")
    except Exception as e:
        print(f"⚠️  JARVIS Memory System Error: {e}")
