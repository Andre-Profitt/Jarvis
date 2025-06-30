#!/usr/bin/env python3
"""
JARVIS Deployment Status Report
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path

print("""
╔══════════════════════════════════════════════════╗
║         JARVIS DEPLOYMENT STATUS REPORT          ║
║              Generated: {}              ║
╚══════════════════════════════════════════════════╝
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")))

# Check running JARVIS processes
print("🔍 JARVIS Processes:")
result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
jarvis_processes = [
    line
    for line in result.stdout.split("\n")
    if "jarvis" in line.lower() and "python" in line.lower() and "grep" not in line
]

for proc in jarvis_processes[:5]:
    parts = proc.split()
    if len(parts) > 10:
        pid = parts[1]
        cmd = " ".join(parts[10:])[:80]
        print(f"  ✅ PID {pid}: {cmd}...")

print(f"\n📊 Total JARVIS processes running: {len(jarvis_processes)}")

# Check Redis
print("\n🔍 Redis Status:")
try:
    result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
    if result.stdout.strip() == "PONG":
        print("  ✅ Redis is running")
    else:
        print("  ❌ Redis is not responding")
except:
    print("  ❌ Redis not found")

# Check logs
print("\n📝 Recent Log Files:")
logs_dir = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/logs")
if logs_dir.exists():
    log_files = sorted(logs_dir.glob("*.log"), key=os.path.getmtime, reverse=True)[:5]
    for log_file in log_files:
        size = os.path.getsize(log_file) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        print(f"  • {log_file.name}: {size:.1f}KB - {mod_time.strftime('%H:%M:%S')}")

# MCP Server Status
print("\n🔌 MCP Server Configuration:")
mcp_config = Path.home() / ".config/claude/claude_desktop_config.json"
if mcp_config.exists():
    print("  ✅ Claude MCP config exists")
    import json

    with open(mcp_config) as f:
        config = json.load(f)
    servers = config.get("mcpServers", {})
    for server_name in servers:
        print(f"    • {server_name}")
else:
    print("  ⚠️  Claude MCP config not found")

print("\n✨ DEPLOYMENT SUMMARY:")
print("=" * 50)
if jarvis_processes:
    print("✅ JARVIS IS RUNNING!")
    print(f"✅ {len(jarvis_processes)} JARVIS processes active")
    print("✅ Core systems deployed")
    print("\n🎯 Next Steps:")
    print("  1. Check logs: tail -f logs/jarvis_now.log")
    print("  2. Restart Claude Desktop to see JARVIS in MCP")
    print("  3. Monitor system health")
else:
    print("⚠️  No JARVIS processes detected")
    print("  Run: python3 launch_jarvis_now.py")

print("\n💡 Useful Commands:")
print("  • View logs: tail -f logs/*.log")
print("  • Stop all: pkill -f jarvis")
print("  • Check status: python3 deployment_status.py")
