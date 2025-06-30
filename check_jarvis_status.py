#!/usr/bin/env python3
"""
JARVIS Status Check - See what's actually working
"""

import os
import sys

print("🔍 JARVIS System Status Check\n")

# Check Python version
print(f"✅ Python version: {sys.version.split()[0]}")

# Check API keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
api_keys = {}
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                if 'API_KEY' in key and value:
                    api_keys[key] = '✅ Configured' if value else '❌ Missing'

print("\n📋 API Keys:")
for key, status in api_keys.items():
    print(f"  {status} {key}")

# Check installed packages
print("\n📦 Required Packages:")
packages = {
    'openai': 'OpenAI GPT-4',
    'google.generativeai': 'Google Gemini',
    'speech_recognition': 'Voice Recognition',
    'pyttsx3': 'Voice Synthesis',
    'flask': 'Web Server',
    'websockets': 'WebSocket Server'
}

for package, name in packages.items():
    try:
        if '.' in package:
            exec(f"import {package}")
        else:
            __import__(package)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - Run: pip3 install {package.split('.')[0]}")

# Check components
print("\n🧩 JARVIS Components Built:")
components = [
    'jarvis_consciousness.py',
    'jarvis_integration.py', 
    'long_term_memory.py',
    'quantum_optimization.py',
    'jarvis_seamless_v2.py',
    'multi_ai_integration.py'
]

for comp in components:
    if os.path.exists(comp):
        print(f"  ✅ {comp}")
    else:
        print(f"  ❌ {comp}")

print("\n📌 What's Left to Do:")
print("1. Install missing packages:")
print("   pip3 install openai google-generativeai")
print("\n2. Run the real system:")
print("   python3 jarvis_real_working.py")
print("\n3. Or run the demo:")
print("   open jarvis-demo-interactive.html")

# Quick test of AI
print("\n🧪 Testing AI Connections...")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say 'JARVIS online' if you can hear me")
    print(f"  ✅ Gemini says: {response.text.strip()}")
except Exception as e:
    print(f"  ❌ Gemini error: {str(e)[:50]}...")
