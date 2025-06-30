#!/usr/bin/env python3
"""
Interactive client for testing the RAG server
This sends proper JSON requests to the server
"""

import json
import subprocess
import sys
from datetime import datetime


def send_to_server(server_process, request):
    """Send a request to the server and get response"""
    # Send request
    server_process.stdin.write(json.dumps(request) + "\n")
    server_process.stdin.flush()

    # Get response
    response = server_process.stdout.readline()
    return json.loads(response)


def main():
    print("🧠 RAG Server Interactive Tester")
    print("=" * 60)

    # Start the server
    print("Starting server...")
    server = subprocess.Popen(
        [sys.executable, "server_simple_working.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    # Skip initial output
    import time

    time.sleep(2)

    while True:
        print("\n📋 What would you like to do?")
        print("1. Store a conversation")
        print("2. Recall memories")
        print("3. Get stats")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            # Store conversation
            user_msg = input("Your message: ")
            assistant_msg = input("Assistant response: ")

            request = {
                "method": "store_conversation",
                "params": {
                    "conversation_id": f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "metadata": {"timestamp": datetime.now().isoformat()},
                },
            }

            response = send_to_server(server, request)
            print(f"Response: {response}")

        elif choice == "2":
            # Recall memories
            query = input("Search for: ")

            request = {
                "method": "recall_memories",
                "params": {"query": query, "top_k": 5},
            }

            response = send_to_server(server, request)
            if "result" in response:
                memories = response["result"].get("memories", [])
                print(f"\nFound {len(memories)} memories:")
                for i, mem in enumerate(memories):
                    print(f"{i+1}. Relevance: {mem.get('relevance', 0):.2f}")
                    conv = mem.get("conversation", {})
                    for msg in conv.get("messages", []):
                        print(f"   {msg.get('role')}: {msg.get('content')}")

        elif choice == "3":
            # Get stats
            request = {"method": "get_memory_stats", "params": {}}
            response = send_to_server(server, request)
            if "result" in response:
                print(f"Stats: {response['result']}")

        elif choice == "4":
            # Exit
            server.terminate()
            break

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
