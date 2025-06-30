#!/usr/bin/env python3
"""
Final JARVIS Deployment Verification
"""
import os
import json
import redis
import subprocess
from pathlib import Path
from datetime import datetime

print("""
╔═══════════════════════════════════════════════════════╗
║         🚀 JARVIS FINAL STATUS CHECK 🚀              ║
╚═══════════════════════════════════════════════════════╝
""")

def check_processes():
    """Check running JARVIS processes"""
    print("📊 Process Status:")
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    jarvis_procs = [l for l in result.stdout.split('\n') if 'jarvis' in l.lower() and 'python' in l.lower()]
    print(f"  ✅ JARVIS Processes: {len(jarvis_procs)}")
    return len(jarvis_procs) > 0

def check_redis():
    """Check Redis status"""
    print("\n💾 Redis Status:")
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        status = r.get("jarvis:status")
        heartbeat = r.get("jarvis:heartbeat")
        print(f"  ✅ Redis: Connected")
        print(f"  ✅ JARVIS Status: {status or 'Not set'}")
        if heartbeat:
            print(f"  ✅ Last Heartbeat: {heartbeat}")
        return True
    except:
        print(f"  ❌ Redis: Not connected")
        return False

def check_multi_ai():
    """Check Multi-AI configuration"""
    print("\n🤖 Multi-AI Configuration:")
    config_path = Path("config/multi_ai_config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  ✅ Config Version: {config['version']}")
        print(f"  ✅ Available Models: {len(config['available_models'])}")
        for model in config['available_models']:
            print(f"     • {model['name']} ({model['max_tokens']:,} tokens)")
        print(f"  ✅ Voice Enabled: {'Yes' if config.get('voice_enabled') else 'No'}")
        return True
    else:
        print(f"  ❌ Configuration not found")
        return False

def check_mcp():
    """Check MCP configuration"""
    print("\n🔌 MCP Integration:")
    mcp_config = Path.home() / ".config/claude/claude_desktop_config.json"
    if mcp_config.exists():
        with open(mcp_config) as f:
            config = json.load(f)
        servers = config.get("mcpServers", {})
        print(f"  ✅ MCP Config: Found")
        print(f"  ✅ Configured Servers: {len(servers)}")
        for server in servers:
            print(f"     • {server}")
        return True
    else:
        print(f"  ❌ MCP config not found")
        return False

def check_api_keys():
    """Check API key availability"""
    print("\n🔑 API Keys:")
    keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GEMINI_API_KEY", 
        "ElevenLabs": "ELEVENLABS_API_KEY"
    }
    available = 0
    for name, env_var in keys.items():
        if os.getenv(env_var):
            print(f"  ✅ {name}: Configured")
            available += 1
        else:
            print(f"  ⚠️  {name}: Not found")
    return available > 0

def main():
    """Run all checks"""
    checks = {
        "Processes": check_processes(),
        "Redis": check_redis(),
        "Multi-AI": check_multi_ai(),
        "MCP": check_mcp(),
        "API Keys": check_api_keys()
    }
    
    # Summary
    print("\n" + "="*50)
    print("📋 DEPLOYMENT SUMMARY")
    print("="*50)
    
    passed = sum(checks.values())
    total = len(checks)
    
    if passed == total:
        print("✅ ALL SYSTEMS OPERATIONAL!")
        print("\n🎉 JARVIS is ready for advanced AI orchestration!")
        print("\n🚀 Next: Restart Claude Desktop to see JARVIS in MCP")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("\nIssues to address:")
        for check, passed in checks.items():
            if not passed:
                print(f"  • Fix {check}")
    
    print("\n💡 Quick Commands:")
    print("  • Test Multi-AI: python3 test_multi_ai.py")
    print("  • View Logs: tail -f logs/*.log")
    print("  • Stop JARVIS: pkill -f jarvis")
    print("  • Restart: python3 LAUNCH-JARVIS-REAL.py")

if __name__ == "__main__":
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    main()
