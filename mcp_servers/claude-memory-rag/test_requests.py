#!/usr/bin/env python3
"""
Quick test for the simple server
"""

import asyncio
import json
from datetime import datetime

# Test data
test_conversation = {
    "method": "store_conversation",
    "params": {
        "conversation_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "messages": [
            {"role": "user", "content": "What is JARVIS capable of?"},
            {
                "role": "assistant",
                "content": "JARVIS has neural resource management with 150x efficiency, self-healing capabilities, and quantum swarm optimization.",
            },
        ],
        "metadata": {"topic": "JARVIS capabilities", "importance": "high"},
    },
}

test_recall = {
    "method": "recall_memories",
    "params": {"query": "JARVIS neural efficiency", "top_k": 3},
}

test_stats = {"method": "get_memory_stats", "params": {}}

# Print test requests
print("Test requests to send to the server:")
print("\n1. Store conversation:")
print(json.dumps(test_conversation))
print("\n2. Recall memories:")
print(json.dumps(test_recall))
print("\n3. Get stats:")
print(json.dumps(test_stats))
print("\n" + "=" * 60)
print("Copy and paste these one at a time into the running server")
